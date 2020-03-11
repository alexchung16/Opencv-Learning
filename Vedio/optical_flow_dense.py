#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : optical_flow_dense.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/11 上午9:42
# @ Software   : PyCharm
#-------------------------------------------------------



import os
import numpy as np
import cv2 as cv


video_dir = '/home/alex/Documents/dataset/opencv_video'

pedestrian_dir = os.path.join(video_dir, 'pedestrian.avi')


if __name__ == "__main__":

    cap = cv.VideoCapture(pedestrian_dir)

    pre_ret, pre_frame = cap.read()

    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(pre_frame)

    hsv[..., 1] =255

    while True:

        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        flow = cv.calcOpticalFlowFarneback(pre_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('optical flow', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', frame)
            cv.imwrite('opticalhsv.png', bgr)
        pre_gray = frame_gray.copy()


