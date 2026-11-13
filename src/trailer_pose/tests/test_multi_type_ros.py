"""Check ROS output and smoothing transitions using real messages, without a master."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import rospy
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointField
    from std_msgs.msg import Header
    import trailer_pose_node as node_module
except ImportError:
    node_module = None


@unittest.skipIf(node_module is None, 'source /opt/ros/noetic/setup.bash 后验证 ROS 接口')
class RosMultiTypeTests(unittest.TestCase):
    def test_auto_type_topics_and_type_change_reset(self):
        estimator = mock.Mock(device='cpu', type_names=['cart_A', 'cart_B'])
        estimator.estimate_points.side_effect = [
            dict(hook_x=-1., hook_y=.1, angle=179., trailer_type='cart_A'),
            dict(hook_x=-1.2, hook_y=.3, angle=-179., trailer_type='cart_A'),
            dict(hook_x=-3., hook_y=-.2, angle=90., trailer_type='cart_B')]
        publishers = {}

        def publisher(topic, message_type, **kwargs):
            publishers[topic] = mock.Mock()
            return publishers[topic]

        params = {'~model_path': '/tmp/multi.pth', '~trailer_type': '', '~multi_frame': 5}
        with mock.patch.object(node_module, 'TrailerPoseEstimator', return_value=estimator) as factory, \
                mock.patch.object(rospy, 'get_param', side_effect=lambda key, default: params.get(key, default)), \
                mock.patch.object(rospy, 'Publisher', side_effect=publisher), \
                mock.patch.object(rospy, 'Subscriber'), mock.patch.object(rospy, 'loginfo'), \
                mock.patch.object(rospy, 'loginfo_throttle'):
            node = node_module.TrailerPoseNode()
            factory.assert_called_once_with('/tmp/multi.pth', '')
            fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, name in enumerate(['x', 'y', 'z', 'intensity'])]
            header = Header(stamp=rospy.Time(10), frame_id='sensor')
            cloud = pc2.create_cloud(header, fields, [[-1., 0., 0., 1.]] * 12)
            node.cloud_callback(cloud)
            node.cloud_callback(cloud)
            second = publishers['/trailer_pose'].publish.call_args[0][0]
            self.assertAlmostEqual(second.pose.position.x, -1.1)
            self.assertAlmostEqual(second.pose.position.y, .2)
            self.assertAlmostEqual(abs(second.pose.orientation.z), 1.)
            self.assertEqual(second.header.frame_id, 'sensor')
            self.assertEqual(second.header.stamp, header.stamp)
            node.cloud_callback(cloud)
        third = publishers['/trailer_pose'].publish.call_args[0][0]
        self.assertAlmostEqual(third.pose.position.x, -3.)
        self.assertEqual(len(node.pose_buffer), 1)
        self.assertEqual(publishers['/trailer_type'].publish.call_args[0][0].data, 'cart_B')

    def test_invalid_smoothing_window_is_rejected(self):
        params = {'~multi_frame': 0}
        with mock.patch.object(node_module, 'TrailerPoseEstimator'), \
                mock.patch.object(rospy, 'get_param', side_effect=lambda key, default: params.get(key, default)):
            with self.assertRaisesRegex(ValueError, 'multi_frame'):
                node_module.TrailerPoseNode()

    def test_known_type_is_forwarded_to_estimator(self):
        params = {'~model_path': '/tmp/multi.pth', '~trailer_type': 'cart_B'}
        with mock.patch.object(node_module, 'TrailerPoseEstimator') as factory, \
                mock.patch.object(rospy, 'get_param', side_effect=lambda key, default: params.get(key, default)), \
                mock.patch.object(rospy, 'Publisher'), mock.patch.object(rospy, 'Subscriber'), \
                mock.patch.object(rospy, 'loginfo'):
            factory.return_value.type_names = ['cart_A', 'cart_B']
            node_module.TrailerPoseNode()
        factory.assert_called_once_with('/tmp/multi.pth', 'cart_B')


if __name__ == '__main__':
    unittest.main()
