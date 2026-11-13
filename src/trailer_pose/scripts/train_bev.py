#!/usr/bin/env python3
"""BEV + CNN trailer pose training and shared utilities."""
import argparse
import csv
import os
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trailer_types import DEFAULT_TYPE, check_type_names, row_type

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LABEL_CSV = os.path.join(PACKAGE_ROOT, 'data', 'dataset_labels.csv')
DEFAULT_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'models', 'best_pose_model.pth')

X_RANGE = (-5.2, 0.0)
Y_RANGE = (-1.2, 1.2)
BEV_W, BEV_H = 256, 128


def read_pcd(f):
    data = []
    with open(f) as fh:
        for line in fh:
            if line.startswith('DATA ascii'):
                break
        for line in fh:
            data.append([float(x) for x in line.split()])
    return np.array(data, dtype=np.float32)


def points_to_bev(pts):
    """Convert Nx4 point cloud to 2x128x256 BEV image."""
    bev = np.zeros((2, BEV_H, BEV_W), dtype=np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ix = ((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * (BEV_W - 1)).astype(np.int32)
    iy = ((y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * (BEV_H - 1)).astype(np.int32)
    valid = (ix >= 0) & (ix < BEV_W) & (iy >= 0) & (iy < BEV_H)
    ix, iy, z = ix[valid], iy[valid], z[valid]
    count = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    zsum = np.zeros((BEV_H, BEV_W), dtype=np.float32)
    np.add.at(count, (iy, ix), 1)
    np.add.at(zsum, (iy, ix), z)
    bev[0] = np.log1p(count) / np.log1p(20)
    mask = count > 0
    bev[1][mask] = zsum[mask] / count[mask]
    return bev


class TrailerDataset(Dataset):
    def __init__(self, label_csv, split, augment=False, type_names=None, return_type=False):
        self.augment = augment
        self.return_type = return_type
        self.label_dir = Path(label_csv).resolve().parent
        with open(label_csv, encoding='utf-8-sig', newline='') as f:
            self.rows = [r for r in csv.DictReader(f) if r['split'] == split]
        self.type_names = check_type_names(type_names or sorted({row_type(r) for r in self.rows})
                                           or [DEFAULT_TYPE])
        unknown = {row_type(r) for r in self.rows} - set(self.type_names)
        if unknown:
            raise ValueError('%s 包含训练词表之外的车型：%s' % (split, sorted(unknown)))
        self.type_ids = [self.type_names.index(row_type(r)) for r in self.rows]
        print('%s: %d frames, types=%s' % (split, len(self.rows), dict(Counter(
            row_type(r) for r in self.rows))))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = Path(r['file'])
        pts = read_pcd(path if path.is_absolute() else self.label_dir / path)
        hx = float(r['hook_x'])
        hy = float(r['hook_y'])
        angle = float(r['angle'])

        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.radians(np.random.uniform(-5, 5))
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array([[c, -s], [s, c]], dtype=np.float32)
                pts[:, :2] = pts[:, :2] @ rot.T
                hx, hy = rot @ np.array([hx, hy])
                angle += np.degrees(theta)
            if np.random.rand() < 0.5:
                pts[:, 1] = -pts[:, 1]
                hy = -hy
                angle = -angle
            if np.random.rand() < 0.5:
                keep = np.random.rand(len(pts)) > 0.2
                pts = pts[keep]
            if np.random.rand() < 0.5:
                pts[:, :3] += np.random.randn(len(pts), 3).astype(np.float32) * 0.01

        bev = points_to_bev(pts)
        hx_norm = (hx - (-2.2)) / 1.3
        hy_norm = hy / 1.0
        ang_rad = np.radians(angle)
        label = np.array([hx_norm, hy_norm, np.sin(ang_rad), np.cos(ang_rad)], dtype=np.float32)
        sample = (torch.from_numpy(bev), torch.from_numpy(label))
        return sample + (self.type_ids[idx],) if self.return_type else sample


def pose_head():
    return nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True),
                         nn.Dropout(0.2), nn.Linear(128, 4))


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = pose_head()

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


class MultiTypePoseNet(nn.Module):
    """Shared BEV features, a type classifier, and one pose head per type."""
    def __init__(self, n_types):
        super().__init__()
        if n_types < 2:
            raise ValueError('多车型网络至少需要两种车型')
        self.features = PoseNet().features
        self.classifier = nn.Linear(256, n_types)
        self.regressors = nn.ModuleList([pose_head() for _ in range(n_types)])

    def forward_all(self, x):
        features = self.features(x)
        logits = self.classifier(features.flatten(1))
        poses = torch.stack([head(features) for head in self.regressors], dim=1)
        return poses, logits

    def forward(self, x, type_ids=None):
        poses, logits = self.forward_all(x)
        if type_ids is None:
            type_ids = logits.argmax(dim=1)
        return poses[torch.arange(len(x), device=x.device), type_ids], logits


def save_pose_checkpoint(model, model_path, type_names):
    type_names = check_type_names(type_names)
    checkpoint = dict(format_version=1,
                      model_kind='multi_type_pose' if isinstance(model, MultiTypePoseNet) else 'pose',
                      type_names=type_names, model_state_dict=model.state_dict())
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pose_model(model_path, device):
    """Load versioned checkpoints or the original single-type state_dict."""
    saved = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in saved:
        if saved.get('format_version') != 1:
            raise ValueError('不支持的模型文件版本')
        type_names = check_type_names(saved['type_names'])
        if saved['model_kind'] == 'multi_type_pose':
            model = MultiTypePoseNet(len(type_names))
        elif saved['model_kind'] == 'pose' and len(type_names) == 1:
            model = PoseNet()
        else:
            raise ValueError('模型结构和车型词表不匹配')
        state = saved['model_state_dict']
    else:
        model, type_names, state = PoseNet(), [DEFAULT_TYPE], saved
    model.load_state_dict(state)
    return model.to(device).eval(), type_names


def validate_requested_type(type_names, trailer_type):
    requested = str(trailer_type).strip() if trailer_type is not None else ''
    if requested and requested not in type_names:
        raise ValueError('模型不支持车型 %s；可选车型：%s' % (requested, ', '.join(type_names)))
    return requested or None


def predict_pose(model, bev, type_names, trailer_type=None):
    """Decode one BEV tensor. Confidence is a raw class probability, not pose quality."""
    requested = validate_requested_type(type_names, trailer_type)
    if len(bev) != 1:
        raise ValueError('predict_pose 每次接收一帧点云')
    with torch.no_grad():
        if isinstance(model, MultiTypePoseNet):
            poses, logits = model.forward_all(bev)
            probabilities = logits.softmax(dim=1)[0]
            predicted_id = int(probabilities.argmax())
            selected_id = type_names.index(requested) if requested else predicted_id
            pred = poses[0, selected_id].cpu().numpy()
            confidence = float(probabilities[selected_id])
        else:
            pred = model(bev).cpu().numpy()[0]
            predicted_id = selected_id = 0
            confidence = None
    return dict(hook_x=float(pred[0] * 1.3 - 2.2), hook_y=float(pred[1]),
                angle=float(np.degrees(np.arctan2(pred[2], pred[3]))),
                trailer_type=type_names[selected_id], predicted_type=type_names[predicted_id],
                type_confidence=confidence)


def pose_metrics(pred, label):
    hx_pred = pred[:, 0] * 1.3 + (-2.2)
    hy_pred = pred[:, 1] * 1.0
    hx_true = label[:, 0] * 1.3 + (-2.2)
    hy_true = label[:, 1] * 1.0
    ang_pred = np.degrees(np.arctan2(pred[:, 2], pred[:, 3]))
    ang_true = np.degrees(np.arctan2(label[:, 2], label[:, 3]))
    ang_err = np.abs((ang_pred - ang_true + 180) % 360 - 180)
    return {
        'hx_mae': float(np.mean(np.abs(hx_pred - hx_true))),
        'hy_mae': float(np.mean(np.abs(hy_pred - hy_true))),
        'ang_mae': float(np.mean(ang_err)),
    }


def evaluate(model, loader, device, type_names=None):
    model.eval()
    all_pred, all_label, all_types, all_predicted_types, all_known = [], [], [], [], []
    type_names = type_names or [DEFAULT_TYPE]
    with torch.no_grad():
        for batch in loader:
            bev, label = batch[:2]
            types = batch[2] if len(batch) == 3 else torch.zeros(len(bev), dtype=torch.long)
            bev = bev.to(device)
            if isinstance(model, MultiTypePoseNet):
                poses, logits = model.forward_all(bev)
                predicted_types = logits.argmax(dim=1)
                indices = torch.arange(len(bev), device=device)
                pred = poses[indices, predicted_types]
                known = poses[indices, types.to(device)]
            else:
                pred = known = model(bev)
                predicted_types = torch.zeros(len(bev), dtype=torch.long)
            all_pred.append(pred.cpu().numpy())
            all_known.append(known.cpu().numpy())
            all_label.append(label.numpy())
            all_types.append(types.numpy())
            all_predicted_types.append(predicted_types.cpu().numpy())
    if not all_pred:
        raise ValueError('验证集为空，请为每种车型补充不同的采集段并重新划分')
    pred, label = np.concatenate(all_pred), np.concatenate(all_label)
    types, predicted_types = np.concatenate(all_types), np.concatenate(all_predicted_types)
    result = pose_metrics(pred, label)
    result['type_accuracy'] = float(np.mean(types == predicted_types))
    result['known_type'] = pose_metrics(np.concatenate(all_known), label)
    result['per_type'] = {}
    for index, name in enumerate(type_names):
        mask = types == index
        if mask.any():
            result['per_type'][name] = dict(pose_metrics(pred[mask], label[mask]),
                                          n_frames=int(mask.sum()),
                                          type_accuracy=float(np.mean(predicted_types[mask] == index)))
    return result


def train(label_csv, model_path, epochs=40, batch_size=32, lr=1e-3, num_workers=2,
          balance_types=True, classification_weight=0.2):
    if epochs < 1 or batch_size < 1 or num_workers < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('epochs、batch_size、lr 必须为正数，num_workers 不能为负数')
    if not np.isfinite(classification_weight) or classification_weight <= 0:
        raise ValueError('classification_weight 必须为正数')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_set = TrailerDataset(label_csv, 'train', augment=True, return_type=True)
    type_names = train_set.type_names
    val_set = TrailerDataset(label_csv, 'val', type_names=type_names, return_type=True)
    if not len(train_set) or not len(val_set):
        raise ValueError('训练集和验证集不能为空；每种车型至少需要 3 个不同采集段才能划分')
    with open(label_csv, encoding='utf-8-sig', newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    all_types = {row_type(row) for row in all_rows}
    if len(all_types) > 1 and DEFAULT_TYPE in all_types:
        raise ValueError('default 是旧标签的未知车型；多车型训练前请补齐实际车型名称')
    if all_types != set(type_names):
        raise ValueError('部分车型没有训练数据，请增加采集段并按车型重新划分')
    if set(val_set.type_ids) != set(range(len(type_names))):
        raise ValueError('部分车型没有验证数据，请为每种车型至少准备 3 个不同采集段')
    sampler = None
    if balance_types and len(type_names) > 1:
        counts = Counter(train_set.type_ids)
        sampler = WeightedRandomSampler([1.0 / counts[index] for index in train_set.type_ids],
                                        len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=num_workers)

    model = (MultiTypePoseNet(len(type_names)) if len(type_names) > 1 else PoseNet()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        losses = []
        for bev, label, types in train_loader:
            bev, label, types = bev.to(device), label.to(device), types.to(device)
            if isinstance(model, MultiTypePoseNet):
                pred, logits = model(bev, types)
                type_loss = F.cross_entropy(logits, types)
            else:
                pred, type_loss = model(bev), 0.0
            pos_loss = F.smooth_l1_loss(pred[:, :2], label[:, :2])
            ang_loss = F.mse_loss(pred[:, 2:], label[:, 2:])
            loss = pos_loss + 0.5 * ang_loss + classification_weight * type_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        metrics = evaluate(model, val_loader, device, type_names)
        val_score = np.mean([m['hx_mae'] + m['hy_mae'] + 0.02 * m['ang_mae']
                             for m in metrics['per_type'].values()])
        print('Epoch %d/%d loss=%.4f val: hx=%.3fm hy=%.3fm ang=%.2f time=%.1fs' % (
            epoch+1, epochs, np.mean(losses), metrics['hx_mae'], metrics['hy_mae'],
            metrics['ang_mae'], time.time()-t0))
        if len(type_names) > 1:
            print('  type accuracy=%.1f%%; known-type pose: hx=%.3fm hy=%.3fm ang=%.2f' % (
                100 * metrics['type_accuracy'], metrics['known_type']['hx_mae'],
                metrics['known_type']['hy_mae'], metrics['known_type']['ang_mae']))
            for name, values in metrics['per_type'].items():
                print('  %s: n=%d hx=%.3fm hy=%.3fm ang=%.2f type_acc=%.1f%%' % (
                    name, values['n_frames'], values['hx_mae'], values['hy_mae'],
                    values['ang_mae'], 100 * values['type_accuracy']))

        if val_score < best_val:
            best_val = val_score
            save_pose_checkpoint(model, model_path, type_names)
            print('  saved best model to', model_path)

    print('Training finished. Best model:', model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train trailer pose model')
    parser.add_argument('--label_csv', default=DEFAULT_LABEL_CSV)
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--no_balance_types', action='store_true', help='关闭多车型均衡采样')
    parser.add_argument('--classification_weight', type=float, default=0.2)
    args = parser.parse_args()
    try:
        train(args.label_csv, args.model_path, args.epochs, args.batch_size, args.lr,
              args.num_workers, not args.no_balance_types, args.classification_weight)
    except (OSError, ValueError) as error:
        parser.exit(1, '训练失败：%s\n' % error)


if __name__ == '__main__':
    main()
