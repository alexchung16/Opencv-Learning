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
import threading
import concurrent.futures


import threading
import cv2

max_connection = 10
pool_sema = threading.BoundedSemaphore(max_connection)
threading_lock = threading.Lock()
threads = []


class VideoCaptureThreading(threading.Thread):

    def __init__(self, func, url, output_dir, num_second_per_clips):
        super(VideoCaptureThreading, self).__init__()
        self.func = func
        self.url = url
        self.output_dir = output_dir
        self.num_second_per_clips = num_second_per_clips

    def run(self):
        threading_lock.acquire()
        self.func(self.url, self.output_dir, self.num_second_per_clips)
        threading_lock.release()


def save_video(video_url, output_dir, num_second_per_clips=1, dst_height=None, dst_width=None):
    """

    Args:
        video_url:
        output_dir:
        dst_height:
        dst_width:
        num_second_per_clips:

    Returns:

    """
    print(video_url)
    pool_sema.acquire() # 控制最大线程数
    print('current threading {}'.format(threading.current_thread().name))

    cap = cv.VideoCapture(video_url)
    # step get video info
    fps = cap.get(cv.CAP_PROP_FPS)
    if dst_height is None:
        dst_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if dst_width is None:
        dst_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # H.264 codec
    output_path = os.path.join(output_dir, str(video_url.split('\\')[-1]) + '.avi')
    out = cv.VideoWriter(filename=output_path, fourcc=fourcc, fps=fps, frameSize=(dst_width, dst_height),
                         isColor=True)
    num_frame = int(fps * num_second_per_clips)
    try:
        for i in range(num_frame):
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            out.write(frame)
        out.release()
        cap.release()
        pool_sema.release()
        # print(f'{video_url} was downloaded...')
    except cv.error as e:
        print(f"Failed to save video, due to {e}")
        raise e


def save_multi_video(video_urls, output_dir, num_second_per_clips):
    """

    Args:
        video_urls:
        output_dir:
        num_second_per_clips:

    Returns:

    """
    for url in video_urls:

        thread = VideoCaptureThreading(save_video, url, output_dir, num_second_per_clips)
        threads.append(thread)

    # start threading
    for thread in threads:
        thread.start() # 开始执行线程

    for thread in threads:
        thread.join()  # 调用join方法， 使主线程在子线程运行完毕后再退出


def main():

    t1 = time.perf_counter()

    video_dir = '../../outputs'
    video_names = os.listdir(video_dir)

    for video_name in video_names:
        save_video(osp.join(video_dir, video_name), output_dir=video_dir)

    t2 = time.perf_counter()

    print(f'Finished in {t2 - t1} seconds')  # Finished in 1.8429918 seconds

    t3 = time.perf_counter()

    video_urls = [osp.join(video_dir, name) for name in video_names]
    save_multi_video(video_urls, video_dir, 1)
    t4 = time.perf_counter()

    print(f'Finished in {t4 - t3} seconds')  # Finished in 1.8429918 seconds


if __name__ == "__main__":
    main()




