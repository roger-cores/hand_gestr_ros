#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import time
import sys

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class DataColl:
    def __init__(self):
        self.output = None
        self.ori = None
        self.blur = None
        self.mask = None

        self.data_methods = ['blur', 'mask', 'ori', 'output']
        self.classes = ['forward', 'backward', 'left', 'right', 'stop', 'fleft', 'fright', 'bleft', 'bright']
        self.numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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

    def save_record(self, class_index):
        cv2.imwrite(self.base_url + self.data_methods[0] + '/' + self.classes[class_index] + '/' + 'img' + `self.numbers[class_index]` + '.jpg', self.blur)
        cv2.imwrite(self.base_url + self.data_methods[1] + '/' + self.classes[class_index] + '/' + 'img' + `self.numbers[class_index]` + '.jpg', self.mask)
        cv2.imwrite(self.base_url + self.data_methods[2] + '/' + self.classes[class_index] + '/' + 'img' + `self.numbers[class_index]` + '.jpg', self.ori)
        cv2.imwrite(self.base_url + self.data_methods[3] + '/' + self.classes[class_index] + '/' + 'img' + `self.numbers[class_index]` + '.jpg', self.output)
        self.numbers[class_index] = self.numbers[class_index] + 1

    def init(self):
        rospy.init_node('data_coll')
        time.sleep(4)

        self.bridge = CvBridge()
        rospy.Subscriber('segment/output', Image, self.output_updated)
        rospy.Subscriber('segment/ori', Image, self.ori_updated)
        rospy.Subscriber('segment/mask', Image, self.mask_updated)
        rospy.Subscriber('segment/blur', Image, self.blur_updated)

        while True:
            if(self.output is not None):
                cv2.imshow('output', self.output)
            if(self.ori is not None):
                cv2.imshow('ori', self.ori)
            if(self.blur is not None):
                cv2.imshow('blur', self.blur)
            if(self.mask is not None):
                cv2.imshow('mask', self.mask)


            k = cv2.waitKey(10)
            if k == 27:
                break;
            elif k == 32:
                self.current_class = None
            elif k == 56:
                self.current_class = self.classes[0]
            elif k == 50:
                self.current_class = self.classes[1]
            elif k == 52:
                self.current_class = self.classes[2]
            elif k == 54:
                self.current_class = self.classes[3]
            elif k == 53:
                self.current_class = self.classes[4]
            elif k == 55:
                self.current_class = self.classes[5]
            elif k == 57:
                self.current_class = self.classes[6]
            elif k == 49:
                self.current_class = self.classes[7]
            elif k == 51:
                self.current_class = self.classes[8]

            rospy.loginfo(self.current_class)


            if self.current_class is not None:
                self.save_record(self.classes.index(self.current_class))

            time.sleep(0.4)





if __name__ == '__main__':
    try:
        DataColl().init()
    except rospy.ROSInterruptException: pass
