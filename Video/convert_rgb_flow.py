#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : convert_rgb_flow.py
# @ Description:
# @ Reference  : https://github.com/deepmind/kinetics-i3d/blob/f1fa01a332179e82cd655e7cd2f2f0c1c04f0c74/preprocess/hmdb_extract_flow.py
#                https://github.com/Rhythmblue/i3d_finetune/issues/2
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/12 下午2:57
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import sys
import threading
import numpy as np
from glob import glob
import cv2 as cv
import tensorflow as tf
from tensorflow.python.platform import app, flags
from multiprocessing import Pool

from Video.flow_visualize import flow_to_color

video_dir = '/home/alex/Documents/dataset/opencv_video'
pedestrian_dir = os.path.join(video_dir, 'pedestrian.avi')

DATA_DIR = '/home/alex/Documents/dataset/HMDB/hmdb51_sta'
SAVE_DIR = '/home/alex/Documents/dataset/HMDB/data/flow'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data')
flags.DEFINE_string('save_dir', SAVE_DIR, 'where to save flow data')
flags.DEFINE_string('format', 'avi', 'what format of data ')
flags.DEFINE_integer('num_threads', 2, 'number of threads')

IMAGE_SIZE = 224



def get_video_length(video_path):
    """
    size of video frame
    :param video_path:
    :return:
    """
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError('Could not open the file.\n{0}'.format(pedestrian_dir))
    CAP_PROP_FRAME_COUNT = cv.CAP_PROP_FRAME_COUNT

    length = int(cap.get(CAP_PROP_FRAME_COUNT))

    return length

def compute_TVL1(video_path, epsilon=1e-5):
    """
    compute optical flow with DualTVL1 algorithm
    :param video_path:
    :return:
    """

    TVL1 = cv.optflow.DualTVL1OpticalFlow_create()

    cap = cv.VideoCapture(video_path)

    pre_ret, pre_frame = cap.read()
    pre_frame = cv.resize(pre_frame, (IMAGE_SIZE, IMAGE_SIZE))
    pre_gray = cv.cvtColor(pre_frame, cv.COLOR_BGR2GRAY)
    # save flow
    flow = []
    video_length = get_video_length(video_path)
    print(video_length)
    for _ in range(video_length - 2):
        cur_ret, cur_frame = cap.read()
        cur_frame = cv.resize(cur_frame, (IMAGE_SIZE, IMAGE_SIZE))
        cur_gray = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)

        cur_flow = TVL1.calc(pre_gray, cur_gray, None)
        assert cur_flow.dtype == np.float32
        # truncate [-20, 20]
        cur_flow[cur_flow > 20] = 20
        cur_flow[cur_flow < -20] =-20
        # scale to [-1, 1]
        max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
        cur_flow = cur_flow / (max_val(cur_flow) + epsilon)
        flow.append(cur_flow)
        pre_gray = cur_gray
        print(cur_flow.shape)
    cap.release()

    return np.array(flow)


def process_video_file(thread_index, filenames):
    """

    :param thread_index:
    :param filename:
    :param save_path:
    :return:
    """
    for filename in filenames:
        flow = compute_TVL1(filename)
        filename, _ = os.path.splitext(filename)
        split_name = filename.split('/')
        save_name = os.path.join(FLAGS.save_dir, split_name[-2], split_name[-1] + '.npy')
        np.save(save_name, flow)

        print('thread {0}: {1} done'.format(thread_index, filename))
        sys.stdout.flush()

def process_dataset():
    """

    :param dataset_dir:
    :return:
    """
    class_names = [file for file in os.listdir(FLAGS.data_dir) if os.path.isdir(os.path.join(FLAGS.data_dir, file))]
    class_dict = {}
    filenames = []
    for index, class_name in enumerate(class_names):
        class_dict[class_name] = index
        for filename in glob(os.path.join(FLAGS.data_dir, class_name, '*.{0}'.format(FLAGS.format))):
            filenames.append(filename)

    return filenames, class_dict

def generate_dir(save_dir, sub_dir):
    """

    :param save_dir:
    :param sub_dir: list
    :return:
    """

    if os.path.exists(save_dir):
        if os.path.isdir((save_dir)):
            for dir in sub_dir:
                if os.path.exists(os.path.join(save_dir, dir)):
                    continue
                else:
                    os.makedirs(os.path.join(save_dir, dir))
    else:
        os.makedirs(save_dir)
        for dir in sub_dir:
            if os.path.exists(os.path.join(save_dir, dir)):
                continue
            else:
                os.makedirs(os.path.join(save_dir, dir))

def execute_flow():
    filenames, class_dict = process_dataset()
    #
    generate_dir(FLAGS.save_dir, class_dict.keys())

    filename_chunk = np.array_split(filenames, FLAGS.num_threads)
    threads = []
    # create a mechanism
    coord  = tf.train.Coordinator()

    for thread_index in range(FLAGS.num_threads):
        args = (thread_index, filename_chunk[thread_index])
        sub_thread = threading.Thread(target=process_video_file, args=args)
        sub_thread.start()
        threads.append(sub_thread)

    coord.join(threads)


def main(argv):
    execute_flow()


if __name__ == "__main__":
    # compute_TVL1(pedestrian_dir)

    # video_dir = '/home/alex/Documents/dataset/opencv_video.text'

    # process_dataset(DATA_DIR)
    # print(os.path.basename(video_dir))

    app.run()
