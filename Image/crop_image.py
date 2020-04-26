#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : crop_image.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/26 下午2:40
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img_path = '../Data/Image/sunflower.jpg'


def crop_resize_img(image, shape):
    """

    :param image:
    :param shape:
    :return:
    """

    img_height, img_width, img_channel = image.shape

    x_min, x_max, y_min, y_max = 0, img_width, 0, img_height

    if img_height > img_width:
        y_min = int(np.ceil((img_height - img_width) / 2))
        y_max = y_min + img_width
    elif img_height < img_width:
        x_min = int(np.ceil((img_width - img_height) / 2))
        x_max = x_min + img_height

    # draw rectangle
    rect_img = image.copy()
    cv.rectangle(rect_img, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 0, 255), thickness=2)

    # rect_img = cv.line(image, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 0, 255), thickness=2)
    crop_img = image[y_min:y_max, x_min:x_max]

    resize_image = cv.resize(crop_img, shape)

    return rect_img, crop_img, resize_image



if __name__ == "__main__":
    image = cv.imread(img_path)

    rect_img, crop_img, resize_image = crop_resize_img(image, shape=(224, 224))
    cv.imshow('rect_image', rect_img)
    cv.imshow('crop_image', crop_img)
    cv.imshow('resize_image', resize_image)
    cv.waitKey()