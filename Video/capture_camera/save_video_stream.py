#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : save_video_stream
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/5/26 15:16
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import queue
import cv2 as cv
import time
import concurrent.futures


output_path = "../../outputs"

# video buffer
video_buffer = queue.Queue()


def get_video_format(cap):
    """
    get video format
    """
    raw_codec_format = int(cap.get(cv.CAP_PROP_FOURCC))
    decoded_codec_format = (chr(raw_codec_format & 0xFF), chr((raw_codec_format & 0xFF00) >> 8),
                            chr((raw_codec_format & 0xFF0000) >> 16), chr((raw_codec_format & 0xFF000000) >> 24))
    return decoded_codec_format


def save_video_to_buffer(cap, num_second_per_clips=None, frame_height=None, frame_width=None):
    """
    save video to buffer
    Args:
        cap:
        num_second_per_clips:
        frame_height:
        frame_width:

    Returns:

    """
    global video_buffer

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

    try:
        while cap.isOpened():
            frames = []

            tmp_stamp = int(time.mktime(time.localtime()))
            for _ in range(num_frame_per_clip):
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                frame = cv.resize(frame, (dst_width, dst_height))
                frames.append(frame)
            video_buffer.put((tmp_stamp, frames))
            break
        # cap.release()
    except cv.error as e:
        print(f"Failed to save video, due to {e}")
        raise e


def save_video(video_stream, save_path, video_name, video, dst_height, dst_width):
    """
    save one clips video to device
    Args:
        stream:
        save_path:
        video_name:
        video:
        dst_height:
        dst_width:

    Returns:

    """
    cap = cv.VideoCapture(video_stream)
    # step get video info
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')  # H.264 codec

    video_path = os.path.join(save_path, str(video_name) + '.avi')
    out = cv.VideoWriter(filename=video_path, fourcc=fourcc, fps=fps, frameSize=(dst_width, dst_height),
                         isColor=True)
    try:
        for i in range(len(video)):
            out.write(video[i])
        out.release()
        cap.release()
    except cv.error as e:
        print(f"Failed to save video, due to {e}")
        raise e


def save_buffer_to_device(video_stream, save_path):
    """
    save buffer video to hardware device
    Args:
        video_stream:
        save_path:

    Returns:

    """
    global video_buffer

    if video_buffer.empty() is False:
        # video_names = sorted(video_buffer.keys())
        # videos = [video_buffer.pop(video_name) for video_name in video_names]
        # dst_height = videos[0][0].shape[0]
        # dst_width = videos[0][0].shape[1]

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for video_name, video in zip(video_names, videos):
        #         executor.submit(save_video, fps, save_path, video_name, video, dst_height, dst_width)

        video_name, video = video_buffer.get()
        dst_height = video[0].shape[0]
        dst_width = video[0].shape[1]
        save_video(video_stream, save_path, video_name, video, dst_height, dst_width)


def save_video_stream(stream_path, output_path, num_second_per_clips=5):
    """
    save video stream
    Args:
        stream_path:
        output_path：
        num_second_per_clips: number second of per video clips

    Returns:

    """

    cap = cv.VideoCapture(stream_path)
    while cap.isOpened():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(save_video_to_buffer, cap, num_second_per_clips)
            executor.submit(save_buffer_to_device, stream_path, output_path)
    cap.release()


def main():

    stream_path = 0
    start = time.perf_counter()
    cap = cv.VideoCapture(stream_path)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')  # Finished in 2.77 second(s)

    save_video_stream(stream_path, output_path, num_second_per_clips=5)


if __name__ == "__main__":
    main()
