#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : rgb_flow.py
# @ Description: pip install opencv-contrib-python
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/12 上午11:01
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import numpy as np
import cv2 as cv
from multiprocessing import Pool
from Video.flow_visualize import flow_to_color


video_dir = '/home/alex/Documents/dataset/opencv_video'

npy_dir = '/home/alex/python_code/kinetics-i3d-master/data'

pedestrian_dir = os.path.join(video_dir, 'pedestrian.avi')
flow_dir = os.path.join(npy_dir, 'v_CricketShot_g04_c01_flow.npy')


IMAGE_SIZE = 256

BOUND = 15

if __name__ == "__main__":

    #
    cap = cv.VideoCapture(pedestrian_dir)

    pre_ret, pre_frame = cap.read()

    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)

    while True:

        cur_ret, cur_frame = cap.read()
        cur_gray = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

        TVL1 = cv.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(pre_gray, cur_gray, None)
        # flow = (flow+BOUND) * (255.0/BOUND)
        # flow = np.round(flow).astype(np.uint8)

        flow_color = flow_to_color(flow, convert_to_bgr=False)

        frame_flow = np.hstack((cur_frame, flow_color))

        cv.imshow('optical flow', frame_flow)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png', cur_frame)
            cv.imwrite('opticalhsv.png', flow_color)
        pre_gray = cur_gray.copy()


    optical_flow = np.load(flow_dir)


