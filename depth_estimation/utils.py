import cv2
import numpy as np


def resize_to_smaller(img1, img2):
    """Returns two resized images such that they match the dimension of the smaller input dimensions."""

    if img1.shape[0] > img2.shape[0]:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    else:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return img1, img2


def resize_to_smallest(imgs):
    """Returns resized images such that they match the dimension of the smallest input dimensions."""

    dim = imgs[0].shape
    for img in imgs:
        if img.shape[0] < dim[0]:
            dim = img.shape

    for i in range(len(imgs)):
        imgs[i] = cv2.resize(imgs[i], (dim[1], dim[0]))

    return imgs


def resize_to_biggest(imgs):
    """Returns resized images such that they match the dimension of the biggest input dimensions."""

    dim = imgs[0].shape
    for img in imgs:
        if img.shape[0] > dim[0]:
            dim = img.shape

    for i in range(len(imgs)):
        imgs[i] = cv2.resize(imgs[i], (dim[1], dim[0]))

    return imgs
