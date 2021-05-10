import os
import pickle

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt

from utils import files_operations


def pre_process_sample(data, phantom_num, sample_gt_path):
    # Normalize signals to 1
    data.a_scan_mat /= numpy.linalg.norm(data.a_scan_mat,
                                         axis=1).reshape(data.a_scan_mat.shape[0], 1)

    # changing sampling rate from 200 to 100 by averaging
    if phantom_num==2:
        data.a_scan_mat = (data.a_scan_mat[:, ::2] + data.a_scan_mat[:, 1::2]) / 2
        data.a_scan_mat = data.a_scan_mat[:, 500:1500]

    data.set_input( data.a_scan_mat)
    cscan_gt = get_sample_gt(sample_gt_path, phantom_num)
    data.set_cscan_gt(cscan_gt)

    return data


def get_sample_gt(path,phantom_num):

    if files_operations.is_file_exist(path):
        sample_gt = files_operations.load_pickle(path)
    else:
        if phantom_num ==1:
            x = np.arange(-100, 100, 1)
            y = np.arange(-100, 100, 1)
            xx, yy = np.meshgrid(x, y, sparse=True)
            rings_radius = np.arange(20, 101, 20)
            thicknesses = [0.523, 0.424, 0.314, 0.224, 0.12]
            sample_gt = np.zeros([200, 200])
            i=1
            for r in rings_radius[::-1]:
                in_circle_inds = (xx ** 2 + yy ** 2) <= (r ** 2)
                sample_gt[in_circle_inds] = thicknesses[-i]
                i += 1
            sample_gt = sample_gt[:185,:]  # crop according to gt
        elif phantom_num==2:
            sample_gt = np.zeros((35, 307))
            inch_to_mm = 25.4
            # calibrate_to_first_sample = 2.54
            thicknesses = np.flip(np.arange(0.03,0.1,0.01)) * inch_to_mm
            left_x = np.arange(6, 300, 38)
            top_y = [5] * 8
            right_x = np.arange(30, 300, 38)
            bottom_y = [30] * 8

            for i in range(len(thicknesses)):
                sample_gt[top_y[i]:bottom_y[i], left_x[i]:right_x[i]] = thicknesses[i]
        else:
            raise ValueError('sample does not exist')

        # plt.imshow(sample_gt)
        # plt.show()
        # cv2.imwrite(save_sample_path+'.jpg',sample_gt*255)
        files_operations.save_to_pickle(sample_gt, path)

    return sample_gt