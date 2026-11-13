#!/usr/bin/env python3
"""Pick one session's pose in Open3D and save a separate session-label CSV."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

FIELDS = [
    'session', 'n_frames', 'n_points', 'hook_x', 'hook_y', 'back_x',
    'back_y', 'centroid_x', 'centroid_y', 'angle', 'length', 'main_points',
]


def compute_pose(picked, length, direct_hook=False):
    picked = np.asarray(picked, dtype=float)
    expected = 3 if direct_hook else 4
    if picked.shape != (expected, 3) or not np.isfinite(picked).all():
        raise ValueError('需要 %d 个有效的三维点' % expected)
    if not np.isfinite(length) or length <= 0:
        raise ValueError('length 必须为正数，单位为米')
    if direct_hook:
        hook, front, rear = picked
    else:
        # Opposite points on the same hook-hole diameter define its center.
        hook = (picked[0] + picked[1]) / 2
        front, rear = picked[2:]
        if np.linalg.norm(picked[0] - picked[1]) < 1e-6:
            raise ValueError('挂钩两侧参考点不能重合')
    direction = front[:2] - rear[:2]
    baseline = np.linalg.norm(direction)
    if baseline < 1e-6:
        raise ValueError('方向点在 XY 平面重合，无法计算角度')
    unit = direction / baseline
    back = hook[:2] - length * unit
    return {
        'hook_x': float(hook[0]), 'hook_y': float(hook[1]),
        'back_x': float(back[0]), 'back_y': float(back[1]),
        'angle': float(np.degrees(np.arctan2(direction[1], direction[0]))),
        'length': float(length),
    }


def pick_indices(cloud, count):
    import open3d as o3d

    vis = o3d.visualization.VisualizerWithEditing()
    if not vis.create_window(window_name='Trailer labels: pick %d points' % count,
                             width=1280, height=800):
        raise RuntimeError('无法创建 Open3D 窗口，请在可访问图形桌面的终端运行')
    try:
        vis.add_geometry(cloud)
        vis.get_render_option().point_size = 3.0
        view = vis.get_view_control()
        view.set_front([0, 0, 1])
        view.set_up([0, 1, 0])
        view.set_lookat(cloud.get_center())
        view.set_zoom(0.7)
        vis.run()
        return list(vis.get_picked_points())
    finally:
        vis.destroy_window()


def write_labels(source, output, row):
    rows = []
    fields = list(FIELDS)
    if source:
        with source.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = list(dict.fromkeys(list(reader.fieldnames or []) + FIELDS))
            rows = list(reader)
    matches = [r for r in rows if r['session'] == row['session']]
    if len(matches) > 1:
        raise ValueError('源 CSV 中存在重复 session，无法安全更新')
    if matches:
        matches[0].update(row)
    else:
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_preview(path, points, picked, pose, session, demo, direct_hook):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    names = ['H', 'A', 'B'] if direct_hook else ['H1', 'H2', 'A', 'B']
    colors = ['crimson', 'darkorange', 'darkgreen', 'purple']
    hook = np.array([pose['hook_x'], pose['hook_y']])
    back = np.array([pose['back_x'], pose['back_y']])
    for panel, ax in enumerate(axes):
        ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.35, color='steelblue')
        for name, point, color in zip(names, picked, colors):
            ax.scatter(*point[:2], s=65, color=color, edgecolor='white', zorder=4)
            offset = (7, -18) if name == 'H2' else (7, 9)
            ax.annotate(name, point[:2], xytext=offset, textcoords='offset points',
                        fontsize=12, weight='bold', color=color)
        if not direct_hook:
            ax.plot(picked[:2, 0], picked[:2, 1], color='crimson', linewidth=1.5)
        ax.scatter(*hook, marker='x', s=110, linewidth=2.5, color='red', zorder=5)
        ax.annotate('hook center', hook, xytext=(-10, -24), ha='right',
                    textcoords='offset points', color='red')
        if panel < 2:
            ax.annotate('', xy=hook, xytext=back,
                        arrowprops={'arrowstyle': '->', 'color': 'red', 'lw': 2})
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
    axes[0].set_xlim(-5.2, 0)
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].set_title('1. Overlay first frames and select hook + direction')
    low = np.minimum(picked[:, :2].min(axis=0), back) - 0.15
    high = np.maximum(picked[:, :2].max(axis=0), hook) + 0.15
    axes[1].set_xlim(low[0], high[0])
    axes[1].set_ylim(low[1], high[1])
    axes[1].set_title('2. Compute center and direction B -> A')
    axes[2].set_xlim(hook[0] - 0.16, hook[0] + 0.16)
    axes[2].set_ylim(hook[1] - 0.14, hook[1] + 0.14)
    axes[2].set_title('3. Hook center detail')
    title = '%s\nhook=(%.4f, %.4f) m | angle=%.4f deg | length=%.4f m' % (
        session, pose['hook_x'], pose['hook_y'], pose['angle'], pose['length'])
    if demo:
        title += '\nDEMO: illustrative picks, no surveyed ground truth'
    fig.suptitle(title, fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session_dir', type=Path)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--length', type=float, required=True,
                        help='挂钩中心到尾部参考中心的已知长度，米')
    parser.add_argument('--source-csv', type=Path,
                        help='复制此 CSV 的其他标签，仅更新所选段')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--direct-hook', action='store_true',
                        help='直接点选 H、A、B；默认点选挂钩相对两侧 H1、H2，再选 A、B')
    parser.add_argument('--indices', nargs='+', type=int,
                        help='重放所载入点云中的点索引，跳过图形窗口')
    parser.add_argument('--demo', action='store_true', help='将回放记录和预览标记为演示')
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames 必须大于 0')
    if not np.isfinite(args.length) or args.length <= 0:
        parser.error('--length 必须为正数')
    if not args.session_dir.is_dir():
        parser.error('session_dir 必须是有效的采集段目录')
    if args.source_csv and args.source_csv.resolve() == args.output.resolve():
        parser.error('--output 必须与 --source-csv 不同，保留源标签')
    record_path = args.output.with_suffix('.picks.json')
    for output in [args.output, record_path, args.preview]:
        if output and output.exists():
            parser.error('输出已存在，请使用新路径：%s' % output)

    import open3d as o3d

    all_files = sorted(args.session_dir.glob('*.pcd'))
    files = all_files[:args.frames]
    if not files:
        parser.error('该段没有 PCD 文件')
    cloud = o3d.geometry.PointCloud()
    for file in files:
        frame = o3d.io.read_point_cloud(str(file))
        if not frame.has_points():
            parser.error('PCD 没有可用点：%s' % file)
        cloud += frame
    cloud.remove_non_finite_points()
    points = np.asarray(cloud.points)
    count = 3 if args.direct_hook else 4
    print('加载 %d/%d 帧，共 %d 个点' % (len(files), len(all_files), len(points)))
    print('顺序：%s；A、B 位于同一条纵向直边，A 靠近挂钩，B 靠近尾部' % (
        'H、A、B' if args.direct_hook else '挂钩同一直径相对两侧 H1、H2、A、B'))
    print('Shift+左键选点；Shift+右键撤销；Q 完成')
    indices = args.indices if args.indices is not None else pick_indices(cloud, count)
    if (len(indices) != count or len(set(indices)) != count
            or any(i < 0 or i >= len(points) for i in indices)):
        parser.error('需要 %d 个不同且有效的点索引' % count)
    picked = points[indices]
    pose = compute_pose(picked, args.length, args.direct_hook)
    row = dict(pose, session=args.session_dir.resolve().name,
               n_frames=len(all_files), n_points=len(points))
    if args.preview:
        save_preview(args.preview, points, picked, pose, row['session'],
                     args.demo, args.direct_hook)
    write_labels(args.source_csv, args.output, row)
    record = {
        'session': row['session'], 'demo': args.demo,
        'selection': 'replayed_indices' if args.indices is not None else 'interactive',
        'direct_hook': args.direct_hook,
        'files': [str(file.resolve()) for file in files],
        'point_order': 'PCD 文件名排序后拼接，去除非有限点',
        'indices': indices, 'picked_xyz': picked.tolist(), 'pose': pose,
        'source_csv': str(args.source_csv.resolve()) if args.source_csv else None,
        'length_source': 'command_line',
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    for key in ['hook_x', 'hook_y', 'angle', 'length']:
        print('%s: %.6f' % (key, pose[key]))
    print('标签：%s\n点选记录：%s' % (args.output, record_path))
    if args.preview:
        print('预览：%s' % args.preview)


if __name__ == '__main__':
    main()
