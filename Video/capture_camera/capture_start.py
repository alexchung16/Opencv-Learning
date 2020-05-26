#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : crop_image.py
# @ Description: opencv-python==1.1.1.26
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/22 下午17:48
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import queue
import cv2 as cv
import time
import concurrent.futures

video_dataset = 'F:/python_code/dataset/video_demo'
rafting_video = os.path.join(video_dataset, 'rafting.avi')
pedestrian_video = os.path.join(video_dataset, 'pedestrian.avi')
time_video = os.path.join(video_dataset, 'time.mp4')


output_path = "../../outputs"
output_video = os.path.join(output_path, 'test.avi')


def get_video_format(cap):
    """
    get video format
    """
    raw_codec_format = int(cap.get(cv.CAP_PROP_FOURCC))
    decoded_codec_format = (chr(raw_codec_format & 0xFF), chr((raw_codec_format & 0xFF00) >> 8),
                            chr((raw_codec_format & 0xFF0000) >> 16), chr((raw_codec_format & 0xFF000000) >> 24))
    return decoded_codec_format


def save_online_video(video_stream, save_path, frame_height=None, frame_width=None, num_second_per_clips=None):
    """
    save online video
    Args:
        video_stream:
        save_path:
        fps:
        frame_height:
        frame_width:
        num_second_per_clips: the time length of per video clips

    Returns:

    """
    cap = cv.VideoCapture(video_stream)
    # step get video info
    fps = cap.get(cv.CAP_PROP_FPS)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    num_frame_per_clip = int(num_second_per_clips * fps)

    if frame_height is None:
        dst_height = height
    else:
        dst_height = frame_height
    if frame_width is None:
        dst_width = width
    else:
        dst_width = frame_width

    # fourcc = cv.VideoWriter_fourcc('a', 'v', 'c', '1')  # avc1 is one of format of h.264
    fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')  # H.264 codec

    out = cv.VideoWriter(filename=save_path, fourcc=fourcc, fps=fps, frameSize=(dst_width, dst_height),
                         isColor=True)
    try:
        while cap.isOpened():
            for _ in range(num_frame_per_clip):
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                frame = cv.resize(frame, (dst_width, dst_height))
                out.write(frame)
            break
    except cv.error as e:
        print(f"Failed to save video, due to {e}")
        raise e
    finally:
        cap.release()


if __name__ == "__main__":
    stream_path = 0
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
    src_codec_format = get_video_format(cap)
    print(src_codec_format)

    #-------------------------- output video config--------------------------
    dst_fps = int(fps/2)
    dst_height = int(height/2)
    dst_width = int(width/2)
    # fourcc = cv.VideoWriter_fourcc(*'X264')
    fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')  # H.264 codec
    # fourcc = cv.VideoWriter_fourcc('a', 'v', 'c', '1')  # avc1 is one of format of h.264

    out = cv.VideoWriter(filename=output_video, fourcc=fourcc, fps=dst_fps, frameSize=(dst_width, dst_height),
                         isColor=True)

    #--------------------------execute write-------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.resize(frame, (dst_width, dst_height))
        out.write(frame)
        cv.imshow("video", frame)
        if cv.waitKey(1) == ord('q'):
            break












