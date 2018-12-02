#!/usr/bin/env python

import time
import cv2
import predict
import numpy as np
import sys
import rospy

CLASSES = ['forward', 'backward', 'left', 'right', 'stop', 'fleft', 'fright', 'bleft', 'bright']
predictor = predict.Predictor(sys.argv[1], sys.argv[2])

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class Exp:
    def __init__(self):
        self.output = None
        self.ori = None
        self.blur = None
        self.mask = None

        self.data_methods = ['blur', 'mask', 'ori', 'output']
        self.classes = ['forward', 'backward', 'left', 'right', 'stop', 'fleft', 'fright', 'bleft', 'bright']
        self.base_url = sys.argv[1]
        self.current_class = None

    def output_updated(self, img_msg):
        self.output = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    def ori_updated(self, img_msg):
        self.ori = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    def blur_updated(self, img_msg):
        self.blur = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    def mask_updated(self, img_msg):
        self.mask = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    def init(self):
        rospy.init_node('exp_predict')
        time.sleep(4)

        self.bridge = CvBridge()
        rospy.Subscriber('segment/output', Image, self.output_updated)
        rospy.Subscriber('segment/ori', Image, self.ori_updated)
        rospy.Subscriber('segment/mask', Image, self.mask_updated)
        rospy.Subscriber('segment/blur', Image, self.blur_updated)

        while True:
            if(self.ori is not None):
                cv2.imshow('ori', self.ori)
                frame = cv2.resize(self.ori, (160, 192))
                frame = np.array(frame)
                frame = np.resize(frame, (160, 192, 1))
                result = predictor.predictClass(frame, CLASSES.__len__())
                rospy.loginfo(CLASSES[result])

            k = cv2.waitKey(10)
            if k == 27:
                break;

            time.sleep(0.4)


if __name__ == '__main__':
    try:
        Exp().init()
    except rospy.ROSInterruptException: pass
