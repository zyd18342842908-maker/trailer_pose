#!/usr/bin/env python3
"""Extract sampled PointCloud2 frames from ROS1 bags, remove ground, save ASCII PCD."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np


@dataclass
class PreprocessConfig:
    frames: int = 20
    sampling: str = 'uniform'
    start_frame: int = 0
    x_range: tuple = (-6.0, 0.0)
    y_range: tuple = (-1.0, 1.0)
    ground_threshold: float = 0.05
    max_ground_tilt: float = 15.0
    ground_z: float = None
    ground_band: float = 0.3
    min_ground_points: int = 30
    ransac_iterations: int = 200
    rotate_z_deg: float = 0.0
    seed: int = 42

    def validate(self):
        if self.frames < 1 or self.start_frame < 0:
            raise ValueError('frames 必须为正整数，start_frame 不能为负数')
        if self.sampling not in ('uniform', 'first'):
            raise ValueError('sampling 必须为 uniform 或 first')
        for name, bounds in [('x_range', self.x_range), ('y_range', self.y_range)]:
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError('%s 必须为两个有限数，且下限小于上限' % name)
        if (not np.isfinite(self.ground_threshold) or self.ground_threshold <= 0 or
                not np.isfinite(self.ground_band) or self.ground_band <= 0):
            raise ValueError('ground_threshold 和 ground_band 必须为有限正数')
        if not np.isfinite(self.max_ground_tilt) or not 0 <= self.max_ground_tilt < 90:
            raise ValueError('max_ground_tilt 必须在 [0, 90) 度内')
        if self.ground_z is not None and not np.isfinite(self.ground_z):
            raise ValueError('ground_z 必须为有限数')
        if self.min_ground_points < 3 or self.ransac_iterations < 1:
            raise ValueError('min_ground_points 至少为 3，ransac_iterations 必须为正整数')
        if not np.isfinite(self.rotate_z_deg) or not 0 <= self.seed < 2 ** 32:
            raise ValueError('rotate_z_deg 必须为有限数，seed 必须在 [0, 2**32) 内')


def sample_indices(total, config):
    available = total - config.start_frame
    if available < config.frames:
        raise ValueError('topic 共 %d 帧，从第 %d 帧起仅剩 %d 帧，无法抽取 %d 帧' % (
            total, config.start_frame, max(0, available), config.frames))
    if config.sampling == 'first':
        return np.arange(config.start_frame, config.start_frame + config.frames)
    return np.linspace(config.start_frame, total - 1, config.frames, dtype=np.int64)


def cloud_to_xyzi(message):
    """Read scalar XYZ/intensity by name, including shuffled fields and padded rows."""
    import sensor_msgs.point_cloud2 as pc2

    names = {field.name for field in message.fields}
    if not {'x', 'y', 'z'}.issubset(names):
        raise ValueError('雷达 PointCloud2 必须包含 x、y、z 字段')
    fields = sorted((field for field in message.fields
                     if field.name in {'x', 'y', 'z', 'intensity'}), key=lambda field: field.offset)
    if any(field.count != 1 for field in fields):
        raise ValueError('x/y/z/intensity 必须是标量字段')
    ordered_names = [field.name for field in fields]
    values = np.asarray(list(pc2.read_points(message, field_names=ordered_names,
                                           skip_nans=False)), dtype=np.float32)
    points = np.zeros((len(values), 4), dtype=np.float32)
    if len(values):
        points[:, :3] = values[:, [ordered_names.index(axis) for axis in ('x', 'y', 'z')]]
        if 'intensity' in ordered_names:
            intensity = values[:, ordered_names.index('intensity')]
            points[:, 3] = np.where(np.isfinite(intensity), intensity, 0.0)
    return points


def fit_ground_plane(xyz, config, rng):
    """Fit a near-horizontal plane to low points (or a known ground-height band)."""
    if config.ground_z is None:
        count = max(config.min_ground_points, int(np.ceil(len(xyz) * 0.3)))
        seeds = xyz[np.argsort(xyz[:, 2])[:count]]
    else:
        seeds = xyz[np.abs(xyz[:, 2] - config.ground_z) <= config.ground_band]
    if len(seeds) < config.min_ground_points:
        raise ValueError('地面候选点不足 %d 个，请检查 ROI 或 ground_z/ground_band' %
                         config.min_ground_points)
    # Bound RANSAC work on dense scans; refine the chosen plane using all seed points.
    sample = seeds if len(seeds) <= 4000 else seeds[rng.choice(len(seeds), 4000, replace=False)]
    min_support = max(3, int(np.ceil(config.min_ground_points * len(sample) / len(seeds))))
    min_normal_z = np.cos(np.radians(config.max_ground_tilt))
    best_plane, best_count, best_error = None, 0, float('inf')
    for _ in range(config.ransac_iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < min_normal_z:
            continue
        offset = -np.dot(normal, a)
        if (config.ground_z is not None and
                abs(-offset / normal[2] - config.ground_z) > config.ground_band):
            continue
        distances = np.abs(sample @ normal + offset)
        inliers = distances <= config.ground_threshold
        support = int(inliers.sum())
        error = float(distances[inliers].mean()) if support else float('inf')
        if support >= min_support and (support > best_count or
                                      (support == best_count and error < best_error)):
            best_plane, best_count, best_error = (normal, offset), support, error
    if best_plane is None:
        raise ValueError('无法拟合近水平地面，请检查 z 轴方向、ground_z 和地面阈值')
    normal, offset = best_plane
    for _ in range(2):
        inliers = seeds[np.abs(seeds @ normal + offset) <= config.ground_threshold]
        if len(inliers) < config.min_ground_points:
            raise ValueError('地面平面支持点不足，请调整 ground_z 或地面阈值')
        center = inliers.mean(axis=0)
        _, singular, vectors = np.linalg.svd(inliers - center, full_matrices=False)
        if singular[1] < 1e-8:
            raise ValueError('地面候选点共线，无法确定地面平面')
        normal = vectors[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -np.dot(normal, center)
        if normal[2] < min_normal_z:
            raise ValueError('拟合地面倾角过大，请检查雷达坐标或 max_ground_tilt')
        if (config.ground_z is not None and
                abs(-offset / normal[2] - config.ground_z) > config.ground_band):
            raise ValueError('拟合地面高度超出 ground_z/ground_band')
    return np.append(normal, offset)


def preprocess_points(points, config, rng):
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points[:, :3]).all(axis=1)
    finite_count = int(valid.sum())
    valid &= ((points[:, 0] >= config.x_range[0]) & (points[:, 0] <= config.x_range[1]) &
              (points[:, 1] >= config.y_range[0]) & (points[:, 1] <= config.y_range[1]))
    cropped = points[valid].copy()
    if not len(cropped):
        raise ValueError('ROI 内没有有效点，请检查 topic、坐标系和 x/y 范围')
    plane = fit_ground_plane(cropped[:, :3], config, rng)
    # Remove ground and points below it; retain objects above the distance threshold.
    keep = cropped[:, :3] @ plane[:3] + plane[3] > config.ground_threshold
    output = cropped[keep]
    if not len(output):
        raise ValueError('去地面后没有剩余点，请检查地面高度、距离阈值和目标位置')
    if config.rotate_z_deg:
        theta = np.radians(config.rotate_z_deg)
        c, s = np.cos(theta), np.sin(theta)
        output[:, :2] = output[:, :2] @ np.array([[c, -s], [s, c]]).T
    stats = dict(input_points=len(points), finite_points=finite_count,
                 roi_points=len(cropped), ground_removed=int((~keep).sum()),
                 output_points=len(output), ground_plane=plane.tolist())
    return output.astype(np.float32), stats


def write_pcd(path, points):
    with Path(path).open('x', encoding='ascii') as stream:
        stream.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION .7\n'
                     'FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
                     'COUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\n'
                     'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n' %
                     (len(points), len(points)))
        np.savetxt(stream, points, fmt='%.9g')


def session_name(bag_path):
    stem = Path(bag_path).stem
    return stem if stem.startswith('record') else 'record_' + stem


def discover_bags(inputs):
    found = set()
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            bags = [item for item in path.rglob('*') if item.is_file() and item.suffix.lower() == '.bag']
            if not bags:
                raise ValueError('目录下没有 .bag 文件：%s' % path)
            found.update(bags)
        elif path.is_file() and path.suffix.lower() == '.bag':
            found.add(path)
        else:
            raise ValueError('请输入 ROS1 .bag 文件或包含 bag 的目录：%s' % path)
    bags = sorted(found)
    names = [session_name(path) for path in bags]
    if len(names) != len(set(names)):
        raise ValueError('多个 bag 生成相同采集段名称，请先将这些 bag 重命名')
    return bags


def process_bag(bag_path, topic, output_dir, config):
    import rosbag

    config.validate()
    bag_path = Path(bag_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    target = output_dir / session_name(bag_path)
    if target.exists():
        raise ValueError('输出采集段已存在，不会覆盖：%s' % target)
    with rosbag.Bag(str(bag_path), 'r') as bag:
        topics = bag.get_type_and_topic_info().topics
        if topic not in topics:
            available = ', '.join('%s (%s)' % (name, info.msg_type) for name, info in topics.items())
            raise ValueError('bag 中没有 topic %s；可用 topic：%s' % (topic, available))
        info = topics[topic]
        if info.msg_type != 'sensor_msgs/PointCloud2':
            raise ValueError('%s 类型为 %s，需要 sensor_msgs/PointCloud2；原始雷达数据需先解码' %
                             (topic, info.msg_type))
        indices = sample_indices(info.message_count, config)
        selected = set(indices.tolist())
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.' + target.name + '.', dir=str(output_dir)))
        manifest = dict(source_bag=str(bag_path), topic=topic, total_topic_frames=info.message_count,
                        config=asdict(config), frames=[])
        rng = np.random.RandomState(config.seed)
        try:
            for index, (_, message, stamp) in enumerate(bag.read_messages(topics=[topic])):
                if index not in selected:
                    continue
                try:
                    points, stats = preprocess_points(cloud_to_xyzi(message), config, rng)
                except ValueError as error:
                    raise ValueError('topic 第 %d 帧：%s' % (index, error)) from error
                name = '%s_%04d.pcd' % (target.name, len(manifest['frames']) + 1)
                write_pcd(temporary / name, points)
                manifest['frames'].append(dict(file=name, source_frame_index=index,
                                               bag_stamp_ns=stamp.to_nsec(),
                                               header_stamp_ns=message.header.stamp.to_nsec(),
                                               source_frame_id=message.header.frame_id, **stats))
                print('  [%d/%d] source=%d points=%d -> %d' % (
                    len(manifest['frames']), config.frames, index, stats['input_points'],
                    stats['output_points']), flush=True)
                if len(manifest['frames']) == config.frames:
                    break
            if len(manifest['frames']) != config.frames:
                raise ValueError('实际读取帧数与 bag 索引不符，未输出不完整采集段')
            with (temporary / 'manifest.json').open('x', encoding='utf-8') as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            if target.exists():
                raise ValueError('输出采集段已存在，不会覆盖：%s' % target)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, help='ROS1 bag 文件或目录（递归搜索）')
    parser.add_argument('--topic', required=True, help='sensor_msgs/PointCloud2 雷达 topic')
    parser.add_argument('--output_dir', required=True, help='PCD 采集段的输出根目录')
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--sampling', choices=['uniform', 'first'], default='uniform')
    parser.add_argument('--start_frame', type=int, default=0, help='起始帧索引，从 0 开始')
    parser.add_argument('--x_range', type=float, nargs=2, default=(-6.0, 0.0), metavar=('MIN', 'MAX'))
    parser.add_argument('--y_range', type=float, nargs=2, default=(-1.0, 1.0), metavar=('MIN', 'MAX'))
    parser.add_argument('--ground_threshold', type=float, default=0.05, help='地面法向距离阈值，米')
    parser.add_argument('--max_ground_tilt', type=float, default=15.0, help='地面相对 XY 平面的最大倾角，度')
    parser.add_argument('--ground_z', type=float, help='已知地面在雷达坐标系原点处的 z，米')
    parser.add_argument('--ground_band', type=float, default=0.3, help='已知地面高度候选带半宽，米')
    parser.add_argument('--min_ground_points', type=int, default=30)
    parser.add_argument('--ransac_iterations', type=int, default=200)
    parser.add_argument('--rotate_z_deg', type=float, default=0.0, help='裁剪去地面后绕 z 轴旋转，度')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    config = PreprocessConfig(**{key: getattr(args, key) for key in PreprocessConfig.__dataclass_fields__})
    try:
        config.validate()
        bags = discover_bags(args.input)
        for bag in bags:
            target = Path(args.output_dir).expanduser() / session_name(bag)
            if target.exists():
                raise ValueError('输出采集段已存在，不会覆盖：%s' % target)
    except (OSError, ValueError) as error:
        parser.exit(1, '预处理失败：%s\n' % error)
    failed = 0
    for bag in bags:
        print('processing:', bag, flush=True)
        try:
            target = process_bag(bag, args.topic, args.output_dir, config)
            print('saved:', target, flush=True)
        except Exception as error:
            failed += 1
            print('预处理失败 [%s]：%s' % (bag, error), file=sys.stderr, flush=True)
    print('完成：成功 %d 个 bag，失败 %d 个 bag' % (len(bags) - failed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
