from skimage.io import imread, imshow
from skimage.feature import hog
import numpy as np
import os
from time import time
import cv2


# This file contains all the necessary definitions used in this project to train, test and predict #

def features(folder, samples=100):
    """add the folder that contains separate folders for separate training images, the definition will label each
    training set with a number. For examples if the training folder contains training images for cats and dogs, the
    function will automatically label cats as 0 and dogs as 1"""
    fd_list = []
    label_list = []
    faulty_images = []
    t0 = time()
    for val, fol in enumerate(os.listdir(folder)):
        for num, img in enumerate(os.listdir(folder + "\\" + fol)):
            path = folder + "\\" + fol + "\\" + img
            try:
                img = cv2.imread(path)
                res = cv2.resize(img, (64, 64))
                hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([5, 60, 100], dtype=np.uint8)
                upper_skin = np.array([28, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                skinMask = cv2.dilate(mask, kernel, iterations=1)
                skin = cv2.bitwise_and(res, res, mask=skinMask)
                bgr = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
                gs = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                fd, h_image = hog(gs, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  visualize=True,
                                  channel_axis=None)
                fd_list.append(fd)
                label_list.append(fol)
                print("Current Folder: {}".format(fol))
                print("Number of image processed: {}".format(num))
            except:
                faulty_images.append(path)
            if num > samples:
                break
    t1 = time()
    time_calc = (t1 - t0) / 60
    total_images = samples * len(os.listdir(folder))
    with open("time_calc.txt", "w") as f:
        f.writelines("Time taken to extract features of {} images: {}".format(total_images, time_calc))
    print("Paths of faulty images: {}".format(faulty_images))
    print("Processing time: {} minutes".format(time_calc))
    return fd_list, label_list


def image_test(image_patch, model):
    """Provide a single image path and the trained model, the function will return the result of the classification"""
    img = image_patch
    res_image = cv2.resize(img, (64, 64))
    hsv = cv2.cvtColor(res_image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([5, 60, 100], dtype=np.uint8)  # [5, 60, 100] works best
    upper_skin = np.array([28, 255, 255], dtype=np.uint8)  # [28, 255, 255] works best
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))  # (8,8) works best
    skinMask = cv2.dilate(mask, kernel, iterations=1)
    skin = cv2.bitwise_and(res_image, res_image, mask=skinMask)
    bgr = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    gs = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image", gs)
    # cv2.waitKey(0)
    fd, h_image = hog(gs, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      visualize=True,
                      channel_axis=None)

    predict = model.predict(fd.reshape(1, -1))

    return predict


# The following function has not been used in this project, this can be skipped, it's a feature to be tested #

def slide_window(image, patch_size):
    """Provide the image as argument and the function will return a list of x,y coordinates for top corner and bottom
    corner of each rectangular patch respectively, use those coordinates to create patches while predicting the image"""
    window = []
    img = imread(image)
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0, h, int(patch_size[1] * 0.5)):
        for x in range(0, w, int(patch_size[0] * 0.5)):
            top_corner = (x, y)
            bottom_corner = (x + patch_size[0], y + patch_size[1])
            window.append([top_corner, bottom_corner])
            if bottom_corner[0] > w:
                break
    return window
