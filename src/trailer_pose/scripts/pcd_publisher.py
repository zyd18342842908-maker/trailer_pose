#!/usr/bin/env python3
"""Publish PCD files as sensor_msgs/PointCloud2 for offline testing."""
import glob
import os
import sys

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bev import read_pcd


def make_cloud_msg(pts, frame_id='lidar'):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
    ]
    return pc2.create_cloud(header, fields, pts)


def main():
    rospy.init_node('pcd_publisher')
    pcd_dir = rospy.get_param('~pcd_dir', '')
    topic = rospy.get_param('~topic', '/points_raw')
    rate_hz = rospy.get_param('~rate', 10)
    frame_id = rospy.get_param('~frame_id', 'lidar')
    loop = rospy.get_param('~loop', True)

    if not pcd_dir or not os.path.isdir(pcd_dir):
        rospy.logerr('Please set valid ~pcd_dir')
        return

    files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    if not files:
        rospy.logerr('No pcd files in %s', pcd_dir)
        return

    pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
    rate = rospy.Rate(rate_hz)
    rospy.loginfo('Publishing %d pcd files from %s to %s', len(files), pcd_dir, topic)

    while not rospy.is_shutdown():
        for f in files:
            if rospy.is_shutdown():
                break
            pts = read_pcd(f)
            msg = make_cloud_msg(pts, frame_id)
            pub.publish(msg)
            rospy.loginfo_throttle(2, 'Published %s', os.path.basename(f))
            rate.sleep()
        if not loop:
            break


if __name__ == '__main__':
    main()
