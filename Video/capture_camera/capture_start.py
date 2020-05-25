#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : crop_image.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/22 下午17:48
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import cv2 as cv
import time
import hashlib

video_dataset = 'F:/python_code/dataset/video_demo'
rafting_video = os.path.join(video_dataset, 'rafting.avi')
pedestrian_video = os.path.join(video_dataset, 'pedestrian.avi')
time_video = os.path.join(video_dataset, 'time.mp4')

output_path = "../../outputs"
output_video = os.path.join(output_path, 'test.mp4')


def get_video_format():
    """
    get video format
    """
    raw_codec_format = int(cap.get(cv.CAP_PROP_FOURCC))
    decoded_codec_format = (chr(raw_codec_format & 0xFF), chr((raw_codec_format & 0xFF00) >> 8),
                            chr((raw_codec_format & 0xFF0000) >> 16), chr((raw_codec_format & 0xFF000000) >> 24))
    return decoded_codec_format


if __name__ == "__main__":
    stream_path = rafting_video
    cap = cv.VideoCapture(stream_path)
    # cap.set(cv.CAP_PROP_FPS, 25)

    #--------------------------- input video info-----------------------------
    # get fps
    fps = cap.get(cv.CAP_PROP_FPS)
    print(fps)
    # get frame count
    count_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print(count_frame)
    # get frame height and width
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    # get codec format
    src_codec_format = get_video_format()
    print(src_codec_format)

    #-------------------------- output video config--------------------------
    dst_fps = fps
    dst_height = int(height)
    dst_width = int(width)
    # fourcc = cv.VideoWriter_fourcc(*'X264')
    fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')  # H.264 codec

    out = cv.VideoWriter(filename=output_video, fourcc=fourcc, fps=dst_fps, frameSize=(dst_height, dst_width),
                         isColor=True)

    #--------------------------execute write-------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        out.write(frame)
        cv.imshow("video", frame)
        if cv.waitKey(1) == ord('q'):
            break

    # local_time = time.localtime()
    # local_stamp = int(time.mktime(local_time))
    # local_time = f'{local_stamp}'
    # print(local_time)








