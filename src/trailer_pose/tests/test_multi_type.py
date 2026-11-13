"""Exercise type labels, recording splits, head selection and weight compatibility."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))
import inference
import prepare_dataset as preparation
import train_bev as training


def write_pcd(path):
    path.write_text('VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                    'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 12\nHEIGHT 1\n'
                    'POINTS 12\nDATA ascii\n' +
                    '\n'.join('%f %f 0 1' % (-1.4 - i * .1, i * .01) for i in range(12)) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MultiTypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def sessions(self, names=('cart_A', 'cart_B'), count=4):
        inputs = []
        for name in names:
            directory = self.root / name
            directory.mkdir()
            rows = []
            for index in range(count):
                session = directory / ('record%03d' % index)
                session.mkdir()
                write_pcd(session / 'a.pcd')
                write_pcd(session / 'b.PCD')
                rows.append(dict(session=session.name, hook_x=-1.5, hook_y=.1, angle=10,
                                 length=1.8, trailer_type=name, session_dir=str(session)))
            source = directory / 'manual.csv'
            write_csv(source, rows)
            inputs.append(source)
        return inputs

    def dataset(self, count=4):
        inputs = self.sessions(count=count)
        output = self.root / 'dataset.csv'
        rows = preparation.prepare_dataset(self.root, inputs, output)
        return rows, output

    def model(self):
        model = training.MultiTypePoseNet(2).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier.bias.copy_(torch.tensor([3., -3.]))
            model.regressors[0][-1].bias.copy_(torch.tensor([0., .1, 0., 1.]))
            model.regressors[1][-1].bias.copy_(torch.tensor([.5, -.2, 1., 0.]))
        return model

    def test_group_split_is_stratified_and_handles_duplicate_basenames(self):
        rows, output = self.dataset()
        self.assertEqual(len(rows), 16)
        group_splits = {}
        for row in rows:
            group_splits.setdefault(row['group_id'], set()).add(row['split'])
        self.assertEqual(len(group_splits), 8)
        self.assertTrue(all(len(values) == 1 for values in group_splits.values()))
        for name in ['cart_A', 'cart_B']:
            self.assertEqual({r['split'] for r in rows if r['trailer_type'] == name},
                             {'train', 'val', 'test'})
        again = preparation.prepare_dataset(self.root, [self.root / 'cart_A/manual.csv',
                                                        self.root / 'cart_B/manual.csv'],
                                             self.root / 'again.csv')
        self.assertEqual([(r['file'], r['split']) for r in rows],
                         [(r['file'], r['split']) for r in again])

    def test_duplicate_group_deduplicates_but_conflicts_are_rejected(self):
        inputs = self.sessions(names=('cart_A',), count=3)
        output = self.root / 'dataset.csv'
        rows = preparation.prepare_dataset(self.root, inputs * 2, output)
        self.assertEqual(len(rows), 6)
        with inputs[0].open() as stream:
            altered = list(csv.DictReader(stream))
        altered[0]['trailer_type'] = 'cart_B'
        conflict = self.root / 'conflict.csv'
        write_csv(conflict, altered)
        with self.assertRaisesRegex(ValueError, '冲突'):
            preparation.prepare_dataset(self.root, inputs + [conflict], output)

    def test_relative_session_dir_and_legacy_default_type(self):
        inputs = self.sessions(names=('cart_A',), count=3)
        with inputs[0].open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.pop('trailer_type')
            row['session_dir'] = row['session']
        write_csv(inputs[0], rows)
        rows = preparation.prepare_dataset(self.root, inputs, self.root / 'legacy.csv')
        self.assertEqual({row['trailer_type'] for row in rows}, {'default'})
        self.assertTrue(all(Path(row['file']).is_file() for row in rows))

    def test_dataset_returns_type_ids_and_preserves_two_value_api(self):
        _, path = self.dataset()
        legacy_api = training.TrailerDataset(path, 'train')
        self.assertEqual(len(legacy_api[0]), 2)
        dataset = training.TrailerDataset(path, 'train', type_names=['cart_B', 'cart_A'],
                                           return_type=True)
        self.assertEqual(dataset[0][2], 1)
        self.assertEqual(dataset[0][0].shape, (2, 128, 256))
        with self.assertRaisesRegex(ValueError, '词表'):
            training.TrailerDataset(path, 'val', type_names=['cart_A'])

    def test_auto_and_specified_type_use_different_pose_heads(self):
        model = self.model()
        bev = torch.zeros(1, 2, 128, 256)
        automatic = training.predict_pose(model, bev, ['cart_A', 'cart_B'])
        specified = training.predict_pose(model, bev, ['cart_A', 'cart_B'], 'cart_B')
        self.assertEqual(automatic['trailer_type'], 'cart_A')
        self.assertAlmostEqual(automatic['hook_x'], -2.2)
        self.assertAlmostEqual(specified['hook_x'], -1.55)
        self.assertAlmostEqual(specified['angle'], 90.)
        self.assertEqual(specified['trailer_type'], 'cart_B')
        self.assertEqual(specified['predicted_type'], 'cart_A')
        self.assertGreater(automatic['type_confidence'], .99)
        self.assertLess(specified['type_confidence'], .01)
        with self.assertRaisesRegex(ValueError, '不支持车型'):
            training.predict_pose(model, bev, ['cart_A', 'cart_B'], 'cart_C')

    def test_checkpoint_round_trip_retains_types_and_head_selection(self):
        path = self.root / 'multi.pth'
        training.save_pose_checkpoint(self.model(), path, ['cart_A', 'cart_B'])
        loaded, names = training.load_pose_model(path, torch.device('cpu'))
        self.assertIsInstance(loaded, training.MultiTypePoseNet)
        self.assertEqual(names, ['cart_A', 'cart_B'])
        result = training.predict_pose(loaded, torch.zeros(1, 2, 128, 256), names, 'cart_B')
        self.assertAlmostEqual(result['angle'], 90.)
        self.assertFalse((self.root / 'multi.pth.tmp').exists())

    def test_validation_reports_auto_and_known_type_errors_separately(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(2, 2, 128, 256),
                                                 torch.tensor([[0., .1, 0., 1.],
                                                               [.5, -.2, 1., 0.]]),
                                                 torch.tensor([0, 1]))
        metrics = training.evaluate(self.model(), torch.utils.data.DataLoader(dataset, batch_size=2),
                                     torch.device('cpu'), ['cart_A', 'cart_B'])
        self.assertEqual(metrics['type_accuracy'], .5)
        self.assertEqual(metrics['known_type']['hx_mae'], 0.)
        self.assertEqual(metrics['known_type']['ang_mae'], 0.)
        self.assertEqual(metrics['per_type']['cart_A']['hx_mae'], 0.)
        self.assertAlmostEqual(metrics['per_type']['cart_B']['hx_mae'], .65, places=5)
        self.assertAlmostEqual(metrics['per_type']['cart_B']['ang_mae'], 90.)

    def test_original_weights_decode_identically(self):
        path = PACKAGE / 'models/best_pose_model.pth'
        old = training.PoseNet().eval()
        old.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        new, names = training.load_pose_model(path, torch.device('cpu'))
        bev = torch.zeros(1, 2, 128, 256)
        with torch.no_grad():
            torch.testing.assert_close(old(bev), new(bev), rtol=0, atol=0)
        self.assertEqual(names, ['default'])
        self.assertIsNone(training.predict_pose(new, bev, names)['type_confidence'])

    def test_single_type_checkpoint_retains_real_name(self):
        path = self.root / 'single.pth'
        training.save_pose_checkpoint(training.PoseNet(), path, ['cart_A'])
        _, names = training.load_pose_model(path, torch.device('cpu'))
        self.assertEqual(names, ['cart_A'])

    def test_multi_frame_never_averages_poses_across_types(self):
        estimator = object.__new__(inference.TrailerPoseEstimator)
        results = [dict(hook_x=-1., hook_y=0., angle=179., trailer_type='A',
                        predicted_type='A', type_confidence=.9),
                   dict(hook_x=-3., hook_y=0., angle=0., trailer_type='B',
                        predicted_type='B', type_confidence=.8),
                   dict(hook_x=-1.2, hook_y=.2, angle=-179., trailer_type='A',
                        predicted_type='A', type_confidence=.7)]
        with mock.patch.object(estimator, 'estimate', side_effect=results):
            result = estimator.estimate_multi_frame(['1', '2', '3'])
        self.assertEqual(result['trailer_type'], 'A')
        self.assertEqual(result['n_frames'], 3)
        self.assertEqual(result['n_frames_used'], 2)
        self.assertAlmostEqual(result['hook_x'], -1.1)
        self.assertAlmostEqual(abs(result['angle']), 180.)
        with self.assertRaisesRegex(ValueError, '至少'):
            estimator.estimate_multi_frame([])

    def test_untyped_rows_cannot_be_a_second_actual_type(self):
        _, path = self.dataset()
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row['trailer_type'] == 'cart_B':
                row['trailer_type'] = 'default'
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, 'default'):
            training.train(path, self.root / 'bad.pth', epochs=1, num_workers=0)
        self.assertFalse((self.root / 'bad.pth').exists())

    def test_two_type_training_and_standalone_inference(self):
        _, path = self.dataset(count=3)
        target = self.root / 'trained.pth'
        training.train(path, target, epochs=1, batch_size=2, num_workers=0)
        estimator = inference.TrailerPoseEstimator(target)
        self.assertEqual(estimator.type_names, ['cart_A', 'cart_B'])
        cloud = self.root / 'cart_B/record000/a.pcd'
        result = estimator.estimate(cloud, 'cart_B')
        self.assertEqual(result['trailer_type'], 'cart_B')
        self.assertTrue(np.isfinite([result['hook_x'], result['hook_y'], result['angle']]).all())


if __name__ == '__main__':
    unittest.main()
