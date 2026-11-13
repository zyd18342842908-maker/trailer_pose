#!/usr/bin/env python3
"""Stratify recording groups by trailer type and expand labels to frames."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trailer_types import DEFAULT_TYPE, row_type

FIELDS = ['file', 'session', 'split', 'hook_x', 'hook_y', 'angle', 'hx_norm', 'hy_norm',
          'sin_angle', 'cos_angle', 'trailer_type', 'session_dir', 'group_id']


def split_sessions(sessions, seed=42):
    groups = defaultdict(list)
    for session in sessions:
        groups[session['trailer_type']].append(session)
    rng = np.random.RandomState(seed)
    for name in sorted(groups):
        group = groups[name]
        indices = rng.permutation(len(group))
        n = len(group)
        if n >= 3:
            n_val = max(1, int(0.15 * n))
            n_train = max(1, min(int(0.7 * n), n - n_val - 1))
        else:
            n_train, n_val = 1, 0
            print('warning: type %s has only %d session(s); cannot cover all three splits' % (name, n))
        for position, index in enumerate(indices):
            group[index]['split'] = ('train' if position < n_train else
                                      'val' if position < n_train + n_val else 'test')
    return sessions


def prepare_dataset(data_dir, session_csvs, output, seed=42):
    data_dir, output = Path(data_dir), Path(output)
    if isinstance(session_csvs, (str, Path)):
        session_csvs = [session_csvs]
    unique = {}
    rejected = 0
    for csv_path in map(Path, session_csvs):
        if output.resolve() == csv_path.resolve():
            raise ValueError('输出文件不能覆盖输入段级标签：%s' % csv_path)
        with csv_path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            required = {'session', 'hook_x', 'hook_y', 'angle', 'length'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('%s 缺少必要段级标签列' % csv_path)
            for row in reader:
                hx, hy, angle, length = [float(row[key]) for key in ['hook_x', 'hook_y', 'angle', 'length']]
                if not np.isfinite([hx, hy, angle, length]).all():
                    raise ValueError('%s 的 %s 含非有限标签' % (csv_path, row['session']))
                if not (-3.5 < hx < -0.9 and abs(hy) < 0.9 and length > 0.5):
                    rejected += 1
                    continue
                if row.get('session_dir', '').strip():
                    directory = Path(row['session_dir']).expanduser()
                    if not directory.is_absolute():
                        directory = csv_path.parent / directory
                else:
                    directory = data_dir / row['session']
                directory = directory.resolve()
                files = sorted(path for path in directory.glob('*')
                               if path.is_file() and path.suffix.lower() == '.pcd')
                if not files:
                    raise ValueError('找不到采集段的 PCD：%s' % directory)
                session = dict(session=row['session'], session_dir=str(directory),
                               group_id=str(directory), trailer_type=row_type(row),
                               hook_x=hx, hook_y=hy, angle=angle, length=length, files=files)
                previous = unique.get(session['group_id'])
                if previous is not None:
                    keys = ['hook_x', 'hook_y', 'angle', 'length', 'trailer_type']
                    if any(previous[key] != session[key] for key in keys):
                        raise ValueError('同一采集段存在冲突标签或车型：%s' % directory)
                    continue
                unique[session['group_id']] = session
    sessions = split_sessions(list(unique.values()), seed)
    if not sessions:
        raise ValueError('筛选后没有有效采集段，请检查标签范围和数据目录')
    rows = []
    for session in sessions:
        hx, hy, angle = session['hook_x'], session['hook_y'], session['angle']
        radians = np.radians(angle)
        for path in session['files']:
            rows.append(dict(file=str(path), session=session['session'], split=session['split'],
                             hook_x=hx, hook_y=hy, angle=angle, hx_norm=(hx + 2.2) / 1.3,
                             hy_norm=hy, sin_angle=np.sin(radians), cos_angle=np.cos(radians),
                             trailer_type=session['trailer_type'], session_dir=session['session_dir'],
                             group_id=session['group_id']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print('valid sessions:', len(sessions), 'filtered:', rejected)
    print('total frames:', len(rows))
    for split in ['train', 'val', 'test']:
        selected = [row for row in rows if row['split'] == split]
        print('%s: %d frames, %d sessions, types=%s' % (
            split, len(selected), len({row['group_id'] for row in selected}),
            dict(Counter(row['trailer_type'] for row in selected))))
    types = {row['trailer_type'] for row in rows}
    if len(types) > 1 and DEFAULT_TYPE in types:
        print('warning: default 表示未补充车型的旧标签，多车型训练前请补齐实际车型')
    print('saved:', output)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='旧 CSV 未提供 session_dir 时的数据根目录')
    parser.add_argument('--session_csv', nargs='+', required=True, help='一个或多个段级标签 CSV')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        prepare_dataset(args.data_dir, args.session_csv, args.output, args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, '生成失败：%s\n' % error)


if __name__ == '__main__':
    main()
