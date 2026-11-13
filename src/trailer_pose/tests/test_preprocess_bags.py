"""Ground/ROI checks and real ROS1 bag extraction without a ROS master."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import preprocess_bags as preprocessing

try:
    import rosbag
    import rospy
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, String
except ImportError:
    rosbag = None


def scene():
    rng = np.random.RandomState(12)
    x, y = np.meshgrid(np.linspace(-6, 0, 30), np.linspace(-1, 1, 20))
    ground = np.column_stack((x.ravel(), y.ravel(),
                              -1 + .04 * x.ravel() + rng.normal(0, .002, x.size),
                              np.ones(x.size)))
    target = np.column_stack((rng.uniform(-5.5, -.5, 60), rng.uniform(-.8, .8, 60),
                              rng.uniform(-.3, .2, 60), np.full(60, 7.)))
    edges = np.array([[-6., -1., .5, 8.], [0., 1., .5, 8.]])
    outside = np.array([[-6.001, 0, 1, 9], [.001, 0, 1, 9],
                        [-2, -1.001, 1, 9], [-2, 1.001, 1, 9],
                        [np.nan, 0, 1, 9], [-2, 0, np.inf, 9]])
    return np.vstack((ground, target, edges, outside)).astype(np.float32), np.vstack((target, edges))


class GroundFilterTests(unittest.TestCase):
    def test_sloped_ground_is_removed_and_roi_boundaries_are_included(self):
        points, expected = scene()
        result, stats = preprocessing.preprocess_points(
            points, preprocessing.PreprocessConfig(), np.random.RandomState(42))
        np.testing.assert_allclose(result, expected, atol=1e-6)
        self.assertEqual(stats['finite_points'], len(points) - 2)
        self.assertEqual(stats['ground_removed'], 600)
        plane = np.array(stats['ground_plane'])
        self.assertLess(abs(plane[0] / plane[2] + .04), .002)
        self.assertLess(abs(-plane[3] / plane[2] + 1), .002)

    def test_rotation_is_applied_after_cropping_and_ground_removal(self):
        points, expected = scene()
        result, _ = preprocessing.preprocess_points(
            points, preprocessing.PreprocessConfig(rotate_z_deg=180), np.random.RandomState(42))
        expected[:, :2] *= -1
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_known_height_avoids_dominant_horizontal_object(self):
        points, _ = scene()
        floor = points[:600]
        rng = np.random.RandomState(1)
        roof = np.column_stack((rng.uniform(-6, 0, 3000), rng.uniform(-1, 1, 3000),
                                np.zeros(3000), np.full(3000, 2)))
        result, stats = preprocessing.preprocess_points(
            np.vstack((floor, roof)), preprocessing.PreprocessConfig(ground_z=-1),
            np.random.RandomState(42))
        self.assertEqual(len(result), 3000)
        self.assertEqual(stats['ground_removed'], 600)

    def test_vertical_wall_is_not_accepted_as_ground(self):
        y, z = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        wall = np.column_stack((-np.ones(y.size), y.ravel(), z.ravel(), np.ones(y.size)))
        with self.assertRaisesRegex(ValueError, '地面'):
            preprocessing.preprocess_points(wall, preprocessing.PreprocessConfig(),
                                             np.random.RandomState(42))

    def test_missing_ground_and_empty_roi_fail_explicitly(self):
        points, _ = scene()
        with self.assertRaisesRegex(ValueError, '候选点不足'):
            preprocessing.preprocess_points(points, preprocessing.PreprocessConfig(ground_z=-5),
                                             np.random.RandomState(42))
        with self.assertRaisesRegex(ValueError, 'ROI'):
            preprocessing.preprocess_points(points, preprocessing.PreprocessConfig(x_range=(20, 30)),
                                             np.random.RandomState(42))

    def test_invalid_configuration_is_rejected(self):
        for changes in [dict(frames=0), dict(start_frame=-1), dict(x_range=(5, 0)),
                        dict(y_range=(-1, np.inf)), dict(ground_z=np.nan),
                        dict(ground_threshold=0), dict(max_ground_tilt=90),
                        dict(min_ground_points=2), dict(seed=-1), dict(rotate_z_deg=np.inf)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                preprocessing.PreprocessConfig(**changes).validate()

    def test_insufficient_frames_are_not_duplicated(self):
        with self.assertRaisesRegex(ValueError, '无法抽取 20 帧'):
            preprocessing.sample_indices(19, preprocessing.PreprocessConfig())
        np.testing.assert_array_equal(preprocessing.sample_indices(
            100, preprocessing.PreprocessConfig(frames=3, sampling='first', start_frame=7)),
            [7, 8, 9])


@unittest.skipIf(rosbag is None, '需要 ROS1 rosbag / sensor_msgs')
class BagExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'processed'

    def tearDown(self):
        self.temp.cleanup()

    def cloud(self, index=0, intensity=True):
        points, _ = scene()
        points[:, 3] = index + 10
        names = ['intensity', 'z', 'x', 'y'] if intensity else ['z', 'x', 'y']
        fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                  for i, name in enumerate(names)]
        columns = [['x', 'y', 'z', 'intensity'].index(name) for name in names]
        return pc2.create_cloud(Header(stamp=rospy.Time(100 + index), frame_id='lidar'),
                                fields, points[:, columns])

    def bag(self, name='test.bag', count=25, intensity=True, broken_frame=None):
        path = self.root / name
        with rosbag.Bag(str(path), 'w') as stream:
            for index in range(count):
                stamp = rospy.Time(100 + index)
                stream.write('/other_topic', String(data='unrelated'), stamp)
                cloud = self.cloud(index, intensity)
                if broken_frame == index:
                    cloud = pc2.create_cloud(cloud.header, cloud.fields, [])
                stream.write('/lidar/points', cloud, stamp)
        return path

    def process(self, path, config=None, topic='/lidar/points'):
        with contextlib.redirect_stdout(io.StringIO()):
            return preprocessing.process_bag(path, topic, self.output,
                                             config or preprocessing.PreprocessConfig())

    def test_real_bag_exports_twenty_distinct_frames_with_provenance(self):
        path = self.bag()
        target = self.process(path)
        files = sorted(target.glob('*.pcd'))
        self.assertEqual(len(files), 20)
        manifest = json.loads((target / 'manifest.json').read_text())
        indices = [row['source_frame_index'] for row in manifest['frames']]
        self.assertEqual(indices, [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 24])
        self.assertEqual(manifest['total_topic_frames'], 25)
        self.assertEqual(manifest['source_bag'], str(path))
        self.assertEqual(manifest['frames'][-1]['header_stamp_ns'], 124000000000)
        self.assertEqual(manifest['frames'][0]['source_frame_id'], 'lidar')
        # Use the actual training reader to check exported PCD compatibility.
        from train_bev import read_pcd
        for file, index in zip(files, indices):
            points = read_pcd(file)
            self.assertEqual(points.shape, (62, 4))
            self.assertTrue(((points[:, 0] >= -6) & (points[:, 0] <= 0)).all())
            self.assertTrue(((points[:, 1] >= -1) & (points[:, 1] <= 1)).all())
            self.assertTrue((points[:, 3] == index + 10).all())
        with self.assertRaisesRegex(ValueError, '不会覆盖'):
            self.process(path)
        self.assertEqual(len(list(target.glob('*.pcd'))), 20)

    def test_xyz_only_clouds_get_zero_intensity(self):
        path = self.bag(count=1, intensity=False)
        target = self.process(path, preprocessing.PreprocessConfig(frames=1))
        from train_bev import read_pcd
        points = read_pcd(next(target.glob('*.pcd')))
        self.assertTrue((points[:, 3] == 0).all())
        self.assertEqual(points.shape, (62, 4))

    def test_nan_intensity_does_not_drop_valid_xyz(self):
        cloud = self.cloud()
        fields = cloud.fields
        cloud = pc2.create_cloud(cloud.header, fields, [[np.nan, .5, 2., .2]])
        points = preprocessing.cloud_to_xyzi(cloud)
        np.testing.assert_allclose(points, [[2., .2, .5, 0]], atol=1e-6)

    def test_failed_frame_leaves_no_partial_session(self):
        path = self.bag(count=20, broken_frame=10)
        with self.assertRaisesRegex(ValueError, '第 10 帧'):
            self.process(path)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_wrong_topic_type_and_short_bag_fail_without_output(self):
        path = self.bag(count=2)
        for topic, message in [('/missing', '可用 topic'), ('/other_topic', '需要 sensor_msgs'),
                               ('/lidar/points', '无法抽取')]:
            with self.subTest(topic=topic), self.assertRaisesRegex(ValueError, message):
                self.process(path, topic=topic)
        self.assertFalse(self.output.exists())

    def test_directory_batch_handles_two_bags(self):
        self.bag(name='one.bag', count=1)
        self.bag(name='two.bag', count=1)
        with contextlib.redirect_stdout(io.StringIO()):
            status = preprocessing.main(['--input', str(self.root), '--topic', '/lidar/points',
                                         '--output_dir', str(self.output), '--frames', '1'])
        self.assertEqual(status, 0)
        self.assertEqual(len(list(self.output.glob('record_*/*.pcd'))), 2)

    def test_duplicate_session_names_are_rejected(self):
        first = self.root / 'first'
        second = self.root / 'second'
        first.mkdir()
        second.mkdir()
        (first / 'same.bag').touch()
        (second / 'same.bag').touch()
        with self.assertRaisesRegex(ValueError, '相同采集段名称'):
            preprocessing.discover_bags([self.root])


if __name__ == '__main__':
    unittest.main()
