#!/usr/bin/env python3
"""Standalone inference script for PCD files."""
import argparse
from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bev import (load_pose_model, predict_pose, read_pcd, points_to_bev,
                       validate_requested_type)

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'models', 'best_pose_model.pth')


class TrailerPoseEstimator:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, trailer_type=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.type_names = load_pose_model(model_path, self.device)
        self.trailer_type = validate_requested_type(self.type_names, trailer_type)

    def estimate_points(self, pts, trailer_type=None):
        bev = points_to_bev(pts)
        bev = torch.from_numpy(bev).unsqueeze(0).to(self.device)
        requested = self.trailer_type if trailer_type is None else trailer_type
        return predict_pose(self.model, bev, self.type_names, requested)

    def estimate(self, pcd_file, trailer_type=None):
        return self.estimate_points(read_pcd(pcd_file), trailer_type)

    def estimate_multi_frame(self, pcd_files, trailer_type=None):
        all_results = [self.estimate(f, trailer_type) for f in pcd_files]
        if not all_results:
            raise ValueError('多帧推理至少需要一个 PCD 文件')
        selected_type = Counter(r['trailer_type'] for r in all_results).most_common(1)[0][0]
        results = [r for r in all_results if r['trailer_type'] == selected_type]
        hx = np.mean([r['hook_x'] for r in results])
        hy = np.mean([r['hook_y'] for r in results])
        sin_mean = np.mean([np.sin(np.radians(r['angle'])) for r in results])
        cos_mean = np.mean([np.cos(np.radians(r['angle'])) for r in results])
        angle = np.degrees(np.arctan2(sin_mean, cos_mean))
        return {
            'hook_x': float(hx),
            'hook_y': float(hy),
            'angle': float(angle),
            'n_frames': len(all_results),
            'n_frames_used': len(results),
            'trailer_type': selected_type,
            'predicted_type': Counter(r['predicted_type'] for r in all_results).most_common(1)[0][0],
            'type_confidence': (float(np.mean([r['type_confidence'] for r in results]))
                                if results[0]['type_confidence'] is not None else None),
        }


def print_result(result):
    print('trailer_type:', result['trailer_type'])
    if result['type_confidence'] is not None:
        print('type_confidence: %.3f (class probability)' % result['type_confidence'])
    if result['predicted_type'] != result['trailer_type']:
        print('predicted_type:', result['predicted_type'], '(using specified type)')
    print('hook_x: %.3f m' % result['hook_x'])
    print('hook_y: %.3f m' % result['hook_y'])
    print('angle:  %.2f deg' % result['angle'])


def main():
    parser = argparse.ArgumentParser(description='Trailer pose inference')
    parser.add_argument('input', help='PCD file or directory')
    parser.add_argument('--multi', action='store_true', help='Use multi-frame average for directory')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--trailer_type', help='已知车型名称；不填则自动识别（多车型模型）')
    args = parser.parse_args()

    try:
        estimator = TrailerPoseEstimator(args.model, args.trailer_type)
        if os.path.isfile(args.input):
            print('file:', args.input)
            print_result(estimator.estimate(args.input))
        elif os.path.isdir(args.input):
            files = sorted(path for path in Path(args.input).iterdir()
                           if path.is_file() and path.suffix.lower() == '.pcd')
            if not files:
                raise ValueError('No pcd files found: %s' % args.input)
            if args.multi:
                result = estimator.estimate_multi_frame(files)
                print('dir:', args.input)
                print('n_frames: %d, n_frames_used: %d' % (result['n_frames'], result['n_frames_used']))
                print_result(result)
            else:
                print('%-40s %16s %10s %10s %10s' % ('file', 'trailer_type', 'hook_x', 'hook_y', 'angle'))
                for path in files:
                    r = estimator.estimate(path)
                    print('%-40s %16s %10.3f %10.3f %10.2f' % (
                        path.name, r['trailer_type'], r['hook_x'], r['hook_y'], r['angle']))
        else:
            raise ValueError('Input not found: %s' % args.input)
    except (OSError, ValueError) as error:
        parser.exit(1, '推理失败：%s\n' % error)


if __name__ == '__main__':
    main()
