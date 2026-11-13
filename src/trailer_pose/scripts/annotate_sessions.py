#!/usr/bin/env python3
"""Interactive session annotation tool.

Usage:
    python3 scripts/annotate_sessions.py --data_dir data/bag_combined --output data/session_analysis.csv

Controls:
    Enter       Keep current values and go to next session
    x,y,angle   Set new values (e.g. -1.2,0.15,5.0) and go to next
    s           Skip current session (keep old values)
    b           Go back to previous session
    q           Save and quit
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_sessions import read_pcd


def load_existing(csv_path):
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path) as f:
        return {r['session']: r for r in csv.DictReader(f)}


def save_csv(csv_path, rows):
    fieldnames = ['session', 'n_frames', 'n_points', 'hook_x', 'hook_y',
                  'back_x', 'back_y', 'centroid_x', 'centroid_y',
                  'angle', 'length', 'main_points']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def show_session(session_dir, label):
    files = sorted(glob.glob(os.path.join(session_dir, '*.pcd')))[:5]
    pts = np.concatenate([read_pcd(f) for f in files])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.4)

    hx = float(label.get('hook_x', 0))
    hy = float(label.get('hook_y', 0))
    angle = float(label.get('angle', 0))
    length = float(label.get('length', 1.5))

    # 画挂钩位置
    ax.scatter([hx], [hy], c='red', s=150, marker='x', linewidths=3, label='hook')

    # 画拖挂方向线
    rad = np.radians(angle)
    bx = hx - length * np.cos(rad)
    by = hy - length * np.sin(rad)
    ax.plot([bx, hx], [by, hy], 'r-', linewidth=2, label='direction')

    ax.set_xlim(-5.2, 0.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('%s\nhook=(%.3f, %.3f) angle=%.2f length=%.2f' % (
        os.path.basename(session_dir), hx, hy, angle, length))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    plt.show(block=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Interactive session annotation')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    existing = load_existing(args.output)
    session_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'record*/')))

    # 初始化所有段
    all_rows = []
    for d in session_dirs:
        name = os.path.basename(d.rstrip('/'))
        n_frames = len(glob.glob(os.path.join(d, '*.pcd')))
        if name in existing:
            row = existing[name]
        else:
            row = {
                'session': name,
                'n_frames': n_frames,
                'n_points': 0,
                'hook_x': -2.0,
                'hook_y': 0.0,
                'back_x': 0.0,
                'back_y': 0.0,
                'centroid_x': 0.0,
                'centroid_y': 0.0,
                'angle': 0.0,
                'length': 1.5,
                'main_points': 0,
            }
        all_rows.append(row)

    print('Loaded %d sessions, %d with existing labels' % (len(all_rows), len(existing)))
    print('Controls: Enter=next, x,y,angle=set, s=skip, b=back, q=save&quit')

    idx = 0
    while 0 <= idx < len(all_rows):
        row = all_rows[idx]
        session_dir = os.path.join(args.data_dir, row['session'])
        fig = show_session(session_dir, row)

        print('\n[%d/%d] %s' % (idx+1, len(all_rows), row['session']))
        print('Current: hook_x=%.3f hook_y=%.3f angle=%.2f length=%.2f' % (
            float(row['hook_x']), float(row['hook_y']), float(row['angle']), float(row['length'])))

        user_input = input('Input> ').strip()
        plt.close(fig)

        if user_input == 'q':
            break
        elif user_input == 'b':
            idx = max(0, idx - 1)
            continue
        elif user_input == 's' or user_input == '':
            idx += 1
            continue
        else:
            try:
                parts = [float(x.strip()) for x in user_input.split(',')]
                if len(parts) == 3:
                    row['hook_x'] = parts[0]
                    row['hook_y'] = parts[1]
                    row['angle'] = parts[2]
                    print('Updated: hook_x=%.3f hook_y=%.3f angle=%.2f' % (
                        parts[0], parts[1], parts[2]))
                    idx += 1
                elif len(parts) == 4:
                    row['hook_x'] = parts[0]
                    row['hook_y'] = parts[1]
                    row['angle'] = parts[2]
                    row['length'] = parts[3]
                    idx += 1
                else:
                    print('Invalid format. Use: x,y,angle or x,y,angle,length')
            except ValueError:
                print('Invalid input. Use: x,y,angle')

    save_csv(args.output, all_rows)
    print('\nSaved %d sessions to %s' % (len(all_rows), args.output))


if __name__ == '__main__':
    main()
