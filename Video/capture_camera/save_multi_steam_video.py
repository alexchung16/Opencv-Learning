#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : save_multi_steam_video
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/21 15:52
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import os.path as osp
import cv2 as cv
import time
import concurrent.futures


import requests
import time
import concurrent.futures


def save_video(video_url, output_dir, dst_height=None, dst_width=None, num_second_per_clips=1):
    """

    Args:
        video_url:
        output_dir:
        dst_height:
        dst_width:
        num_second_per_clips:

    Returns:

    """
    cap = cv.VideoCapture(video_url)
    # step get video info
    fps = cap.get(cv.CAP_PROP_FPS)
    if dst_height is None:
        dst_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if dst_width is None:
        dst_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fourcc = cv.VideoWriter_fourcc('X', '2', '6', '4')  # H.264 codec
    output_path = os.path.join(output_dir, str(video_url.split('/')[-1]))
    out = cv.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(dst_width, dst_height),
                         isColor=True)
    try:
        for i in range(int(fps * num_second_per_clips)):
            ret, frame = cap.read()
            out.write(frame)
        out.release()
        cap.release()
        print(f'{video_url} was downloaded...')
    except cv.error as e:
        print(f"Failed to save video, due to {e}")
        raise e


def save_multi_video(video_urls, output_dir, num_second_per_clips):

    with concurrent.futures.ThreadPoolExecutor() as executor:
        output_dir = [output_dir] * len(video_urls)
        num_second_per_clips = [num_second_per_clips] * len(video_urls)
        executor.map(save_video, video_urls, output_dir, num_second_per_clips)


def main():

    t1 = time.perf_counter()

    video_dir = '../../outputs'
    video_names = os.listdir(video_dir)

    for video_name in video_names:
        save_video(osp.join(video_dir, video_name), output_dir=video_dir)

    t2 = time.perf_counter()

    print(f'Finished in {t2 - t1} seconds')  # Finished in 1.8429918 seconds

    t3 = time.perf_counter()

    video_dir = '../../outputs'
    video_names = os.listdir(video_dir)

    video_urls = [osp.join(video_dir, name) for name in video_names]
    save_multi_video(video_urls, video_dir, 1)
    t4 = time.perf_counter()

    print(f'Finished in {t4 - t3} seconds')  # Finished in 1.8429918 seconds


if __name__ == "__main__":
    main()