#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : rgb_flow_i3d.py
# @ Description: pip install opencv-contrib-python
# @ Reference  : https://github.com/deepmind/kinetics-i3d
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
from Video.optical_flow.flow_visualize import flow_to_color


video_dir = '/home/alex/Documents/dataset/opencv_video'
basketball_dir = '/home/alex/Documents/dataset/UCF_101/Basketball'
npy_dir = '/home/alex/python_code/kinetics-i3d-master/data'
img_dir = '../Data/Video'

pedestrian_path = os.path.join(video_dir, 'pedestrian.avi')

skateboard_path = os.path.join(basketball_dir, 'v_Basketball_g01_c01.avi')


traffic_path = os.path.join(img_dir, 'traffic.mp4')

flow_dir = os.path.join(npy_dir, 'v_CricketShot_g04_c01_flow.npy')


IMAGE_SIZE = 256

BOUND = 15


def flow_to_rgb(flow):
    """
    reference https://github.com/deepmind/kinetics-i3d
    convert optical flow to rgb
    :param flow_uv:
    :return:
    """

    # step 1 convert scale to [-1, 1]
    flow[flow > 20] = 20
    flow[flow < -20] = -20
    max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
    flow = flow / (max_val(flow) + 1e-5)
    # step 2  convert scale to [0, 1]
    flow += 1.
    flow /= 2.
    # step generate three channel array
    flow_image = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    flow_image[..., 0:2] = flow
    flow_image[..., 2] = 0.5

    return flow_image


if __name__ == "__main__":

    #
    cap = cv.VideoCapture(pedestrian_path)

    video_fps = cap.get(propId=cv.CAP_PROP_FPS)  # fps
    video_frames = cap.get(propId = cv.CAP_PROP_FRAME_COUNT) # frames

    print('fps: {0}'.format(video_fps))
    print('frames: {0}'.format(video_frames))

    pre_ret, pre_frame = cap.read()

    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)

    while True:

        cur_ret, cur_frame = cap.read()
        cur_gray = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

        TVL1 = cv.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(pre_gray, cur_gray, None)
        # flow = (flow+BOUND) * (255.0/BOUND)
        # flow = np.round(flow).astype(np.uint8)

        #-------------------# visulize method 1-------------------
        frame_flow = flow_to_color(flow)
        # parallel display rgb and optical flow frame
        rgb_flow = np.hstack((cur_frame, frame_flow))

        # -------------------# visulize method 2-------------------
        # frame_flow = flow_to_rgb(flow)
        # # transfer pixel size to equal scale
        # # parallel display rgb and optical flow frame
        # rgb_flow = np.hstack((cur_frame/255., frame_flow))

        cv.imshow('optical flow', rgb_flow)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    # optical_flow = np.load(flow_dir)


