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
import cv2 as cv


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


def convert_video_format(video_stream, output_path, dst_height=None, dst_width=None, dst_fps=None,  is_show=False):
    """
    convert video format
    Args:
        video_stream:
        output_path:
        dst_height:
        dst_width:
        dst_fps:
        is_show:

    Returns:

    """
    cap = cv.VideoCapture(video_stream)
    # step get video info
    fps = cap.get(cv.CAP_PROP_FPS)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    if dst_height is None:
        dst_height = height
    else:
        dst_height = dst_height
    if dst_width is None:
        dst_width = width
    else:
        dst_width = dst_width

    if dst_fps is None:
        dst_fps = fps

    # fourcc = cv.VideoWriter_fourcc('a', 'v', 'c', '1')  # avc1 is one of format of h.264
    # fourcc = cv.VideoWriter_fourcc(*'X264')
    fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')  # H.264 codec
    out = cv.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(dst_width, dst_height),
                         isColor=True)
    try:
        show_time_per_frame = int(1000 / dst_fps)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv.resize(frame, (dst_width, dst_height))
            out.write(frame)
            if is_show:
                cv.imshow("video", frame)
                if cv.waitKey(show_time_per_frame) == ord('q'):
                    break
        cap.release()
    except cv.error as e:
        print(f"Failed to save video, due to {e}")
        raise e


def main():
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
    src_codec_format = get_video_format(cap)
    print(src_codec_format)

    #-------------------------- output video config--------------------------
    dst_fps = fps
    dst_height = int(height)
    dst_width = int(width)
    # fourcc = cv.VideoWriter_fourcc(*'X264')
    fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')  # H.264 codec
    # fourcc = cv.VideoWriter_fourcc('a', 'v', 'c', '1')  # avc1 is one of format of h.264

    out = cv.VideoWriter(filename=output_video, fourcc=fourcc, fps=dst_fps, frameSize=(dst_width, dst_height))

    wait_time = int(1000 / dst_fps)
    #--------------------------execute write-------------------------------
    while cap.isOpened():
         ret, frame = cap.read()
         if not ret:
              print("Can't receive frame (stream end?). Exiting ...")
              break
         frame = cv.resize(frame, (dst_width, dst_height))
         out.write(frame)
         cv.imshow("video", frame)
         if cv.waitKey(wait_time) == ord('q'):
              break


if __name__ == "__main__":
    video_stream = rafting_video
    #----------------------get input video info------------------------------
    cap = cv.VideoCapture(video_stream)
    # cap.set(cv.CAP_PROP_FPS, 25)
    # get fps(Frame per second)
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f'Video FPS: {fps}')
    # get frame count
    count_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print(f'Number frames of video: {count_frame}')
    # get frame height and width
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    print(f'Frame height: {height}')
    print(f'Frame width: {width}')
    # get codec format
    src_codec_format = get_video_format(cap)
    print(f'Video codec format: {src_codec_format}')
    #-------------------convert video fomat
    convert_video_format(video_stream=video_stream, output_path=output_video, is_show=True)

    #----------------get target video codec format-------------------------
    cap_output = cv.VideoCapture(output_video)
    # get codec format
    dst_codec_format = get_video_format(cap_output)
    print(f'Target video codec format: {dst_codec_format}')
    print('Done!')













