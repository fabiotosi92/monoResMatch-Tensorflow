import numpy as np
import tensorflow as tf
import cv2


def colormap_jet(img):
    color_image = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_JET)
    return color_image


def color_disparity(disparity):
    with tf.variable_scope('color_disparity'):
        batch_size = disparity.shape[0]
        color_maps = []
        for i in range(batch_size):
            color_disp = tf.py_func(colormap_jet, [-disparity[i]], tf.uint8)
            color_maps.append(color_disp)
        color_batch = tf.stack(color_maps, axis=0)
        return color_batch


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


