#!/usr/bin/env python3
"""ROS1 node: subscribe PointCloud2, publish trailer PoseStamped."""
import os
import sys
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf.transformations import quaternion_from_euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import TrailerPoseEstimator

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'models', 'best_pose_model.pth')


class TrailerPoseNode:
    def __init__(self):
        model_path = rospy.get_param('~model_path', DEFAULT_MODEL_PATH)
        self.estimator = TrailerPoseEstimator(model_path, rospy.get_param('~trailer_type', ''))

        self.multi_frame = rospy.get_param('~multi_frame', 5)
        if not isinstance(self.multi_frame, int) or self.multi_frame < 1:
            raise ValueError('~multi_frame 必须为正整数')
        self.pose_buffer = []
        self.buffer_type = None
        self.buffer_lock = threading.Lock()

        self.frame_id = rospy.get_param('~frame_id', 'lidar')
        cloud_topic = rospy.get_param('~cloud_topic', '/points_raw')
        pose_topic = rospy.get_param('~pose_topic', '/trailer_pose')
        type_topic = rospy.get_param('~type_topic', '/trailer_type')

        self.pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=10)
        self.type_pub = rospy.Publisher(type_topic, String, queue_size=10)
        self.sub = rospy.Subscriber(cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)
        rospy.loginfo('Trailer pose node started. cloud=%s pose=%s type=%s multi_frame=%d device=%s types=%s',
                      cloud_topic, pose_topic, type_topic, self.multi_frame, self.estimator.device,
                      ','.join(self.estimator.type_names))

    def cloud_callback(self, msg):
        points = []
        for p in pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True):
            points.append([p[0], p[1], p[2], p[3]])
        if len(points) < 10:
            rospy.logwarn_throttle(5, 'Too few points: %d', len(points))
            return
        pts = np.array(points, dtype=np.float32)

        result = self.estimator.estimate_points(pts)
        hx, hy, angle = result['hook_x'], result['hook_y'], result['angle']
        trailer_type = result['trailer_type']

        with self.buffer_lock:
            if trailer_type != self.buffer_type:
                self.pose_buffer.clear()
                self.buffer_type = trailer_type
            self.pose_buffer.append((hx, hy, angle))
            if len(self.pose_buffer) > self.multi_frame:
                self.pose_buffer.pop(0)
            buffer = list(self.pose_buffer)

        hx_mean = np.mean([p[0] for p in buffer])
        hy_mean = np.mean([p[1] for p in buffer])
        sin_mean = np.mean([np.sin(np.radians(p[2])) for p in buffer])
        cos_mean = np.mean([np.cos(np.radians(p[2])) for p in buffer])
        angle_mean = np.degrees(np.arctan2(sin_mean, cos_mean))

        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.frame_id
        pose_msg.pose.position.x = hx_mean
        pose_msg.pose.position.y = hy_mean
        pose_msg.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, np.radians(angle_mean))
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        self.pub.publish(pose_msg)
        self.type_pub.publish(String(data=trailer_type))

        rospy.loginfo_throttle(1, 'Pose: type=%s x=%.3f y=%.3f angle=%.2f (buffer=%d)',
                               trailer_type, hx_mean, hy_mean, angle_mean, len(buffer))


def main():
    rospy.init_node('trailer_pose_estimator')
    TrailerPoseNode()
    rospy.spin()


if __name__ == '__main__':
    main()
