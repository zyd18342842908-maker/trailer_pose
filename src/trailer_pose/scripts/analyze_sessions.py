#!/usr/bin/env python3
"""Generate pseudo labels for each recording session."""
import argparse
import csv
import glob
import os

import numpy as np
from scipy import ndimage


def read_pcd(f):
    data = []
    with open(f) as fh:
        for line in fh:
            if line.startswith('DATA ascii'):
                break
        for line in fh:
            data.append([float(x) for x in line.split()])
    return np.array(data, dtype=np.float32)


def analyze_session(d, max_frames=5):
    files = sorted(glob.glob(os.path.join(d, '*.pcd')))[:max_frames]
    if not files:
        return None
    pts = np.concatenate([read_pcd(f) for f in files])
    pts = pts[pts[:, 0] < -0.8]
    if len(pts) < 50:
        return None

    res = 0.05
    x_min, x_max = -5.2, -0.7
    y_min, y_max = -1.2, 1.2
    nx = int((x_max - x_min) / res) + 1
    ny = int((y_max - y_min) / res) + 1
    ix = ((pts[:, 0] - x_min) / res).astype(int)
    iy = ((pts[:, 1] - y_min) / res).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy = ix[valid], iy[valid]
    grid = np.zeros((nx, ny), dtype=int)
    np.add.at(grid, (ix, iy), 1)
    dense = grid >= 2
    if dense.sum() < 5:
        return None
    labels, n = ndimage.label(dense, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndimage.sum(dense, labels, range(1, n + 1))
    main_label = np.argmax(sizes) + 1
    main_mask = labels == main_label
    point_in_main = np.zeros(len(pts), dtype=bool)
    point_in_main[valid] = main_mask[ix, iy]
    main_pts = pts[point_in_main]
    if len(main_pts) < 30:
        return None

    centroid = main_pts[:, :2].mean(axis=0)

    thresh = np.percentile(main_pts[:, 0], 98)
    front = main_pts[main_pts[:, 0] >= thresh]
    hook_x, hook_y = front[:, 0].mean(), front[:, 1].mean()

    thresh_back = np.percentile(main_pts[:, 0], 2)
    back = main_pts[main_pts[:, 0] <= thresh_back]
    back_x, back_y = back[:, 0].mean(), back[:, 1].mean()

    dx = hook_x - back_x
    dy = hook_y - back_y
    angle = np.degrees(np.arctan2(dy, dx))
    length = np.sqrt(dx**2 + dy**2)

    return {
        'session': os.path.basename(d.rstrip('/')),
        'n_frames': len(glob.glob(os.path.join(d, '*.pcd'))),
        'n_points': len(pts),
        'hook_x': hook_x,
        'hook_y': hook_y,
        'back_x': back_x,
        'back_y': back_y,
        'centroid_x': centroid[0],
        'centroid_y': centroid[1],
        'angle': angle,
        'length': length,
        'main_points': len(main_pts),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo labels from PCD sessions')
    parser.add_argument('--data_dir', required=True, help='Directory containing record*/ subdirectories')
    parser.add_argument('--output', required=True, help='Output CSV path')
    args = parser.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.data_dir, 'record*/')))
    results = []
    for i, d in enumerate(dirs):
        r = analyze_session(d)
        if r:
            results.append(r)
        if (i + 1) % 100 == 0:
            print('processed %d/%d' % (i+1, len(dirs)))
    print('total valid: %d/%d' % (len(results), len(dirs)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print('saved:', args.output)


if __name__ == '__main__':
    main()
