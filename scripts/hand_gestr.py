#!/usr/bin/env python

import cv2
import numpy as np
import copy
import math
import sys
import time
import rospy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError


class HandGestr:
    def __init__(self):
        # Params
        self.cap_region_x_begin=0.5  # start point/total width
        self.cap_region_y_end=0.8  # start point/total width
        self.threshold = 60  #  BINARY threshold
        self.blurValue = 31  # GaussianBlur parameter
        self.bgSubThreshold = 50
        self.learningRate = 0

        # Variables
        self.isBgCaptured = 0   # bool, whether the background captured
        self.triggerSwitch = False  # if true, keyborad simulator works

        self.bridge = CvBridge()
        self.output_publisher = rospy.Publisher('segment/output', Image, queue_size=10)
        self.mask_publisher = rospy.Publisher('segment/mask', Image, queue_size=10)
        self.blur_publisher = rospy.Publisher('segment/blur', Image, queue_size=10)
        self.ori_publisher = rospy.Publisher('segment/ori', Image, queue_size=10)



    def printThreshold(self, thr):
        print("! Changed threshold to "+str(thr))


    def removeBG(self, frame):
        fgmask = self.bgModel.apply(frame,learningRate=self.learningRate)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res


    def calculateFingers(self, res, drawing):  # -> finished bool, cnt: finger count
        #  convexity defect
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            if type(defects) != type(None):  # avoid crashing.   (BUG not found)

                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                return True, cnt
        return False, 0

    def init(self):
        rospy.init_node('hand_gestr')
        time.sleep(2)

        # Camera
        camera = cv2.VideoCapture(0)
        camera.set(10,200)
        cv2.namedWindow('trackbar')
        cv2.createTrackbar('trh1', 'trackbar', self.threshold, 100, self.printThreshold)


        while camera.isOpened():
            ret, frame = camera.read()
            self.threshold = cv2.getTrackbarPos('trh1', 'trackbar')
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
            frame = cv2.flip(frame, 1)  # flip the frame horizontally
            cv2.rectangle(frame, (int(self.cap_region_x_begin * frame.shape[1]), 0),
                         (frame.shape[1], int(self.cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
            cv2.imshow('original', frame)

            #  Main operation
            if self.isBgCaptured == 1:  # this part wont run until background captured
                img = self.removeBG(frame)
                img = img[0:int(self.cap_region_y_end * frame.shape[0]),
                            int(self.cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
                image_message = self.bridge.cv2_to_imgmsg(img, encoding="passthrough")
                self.mask_publisher.publish(image_message)

                # convert the image into binary image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 0)
                image_message = self.bridge.cv2_to_imgmsg(blur, encoding="passthrough")
                self.blur_publisher.publish(image_message)
                ret, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)
                image_message = self.bridge.cv2_to_imgmsg(thresh, encoding="passthrough")
                self.ori_publisher.publish(image_message)

                # get the coutours
                thresh1 = copy.deepcopy(thresh)
                _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                length = len(contours)
                maxArea = -1
                if length > 0:
                    for i in range(length):  # find the biggest contour (according to area)
                        temp = contours[i]
                        area = cv2.contourArea(temp)
                        if area > maxArea:
                            maxArea = area
                            ci = i

                    res = contours[ci]
                    hull = cv2.convexHull(res)
                    drawing = np.zeros(img.shape, np.uint8)
                    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                    isFinishCal,cnt = self.calculateFingers(res,drawing)
                    if self.triggerSwitch is True:
                        if isFinishCal is True and cnt <= 2:
                            print (cnt)
                            #app('System Events').keystroke(' ')  # simulate pressing blank space


                # cv2.imshow('output', drawing)
                image_message = self.bridge.cv2_to_imgmsg(drawing, encoding="passthrough")
                self.output_publisher.publish(image_message)

            # Keyboard OP
            k = cv2.waitKey(10)
            if k == 27:  # press ESC to exit
                break
            elif k == ord('b'):  # press 'b' to capture the background
                self.bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)
                self.isBgCaptured = 1
                print( '!!!Background Captured!!!')
            elif k == ord('r'):  # press 'r' to reset the background
                self.bgModel = None
                self.triggerSwitch = False
                self.isBgCaptured = 0
                print ('!!!Reset BackGround!!!')
            elif k == ord('n'):
                self.triggerSwitch = True
                print ('!!!Trigger On!!!')

if __name__ == '__main__':
    try:
        HandGestr().init()
    except rospy.ROSInterruptException: pass
