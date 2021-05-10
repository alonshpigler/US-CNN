import os
from copy import deepcopy

import numpy as np
import scipy
from matplotlib import pyplot as plt
from utils import files_operations
from utils.HdfDataHandler import DispType


def eval_sample(a_scan_mat_pred,
                test_data,
                phantom_num,
                material_velocity,
                vis_args,
                layer_eval_type='first_two',
                metrics=['mae'],  # options: {'mae', 'rmse',relative_mae',relative_mae'}
                translate_time_to_thickness_from_data = False,
                provide_extra_res=False,
                timeframe_units=False,
                visualize =True):

    thickness_vals, vis_args = get_sample_args(phantom_num, vis_args)
    pred_c_scan = get_c_scan_pred(a_scan_mat_pred, test_data,layer_eval_type)
    post_cscan = post_process_c_scan(pred_c_scan, phantom_num)
    cscan_target = test_data.c_scan_gt
    thickness_areas = get_sample_areas(phantom_num)
    if timeframe_units:
        cscan_target = convert_distance_to_time(post_cscan, material_velocity, translate_time_to_thickness_from_data, cscan_target, thickness_areas)
    else:
        post_cscan = convert_time_to_distance(post_cscan, material_velocity, translate_time_to_thickness_from_data, cscan_target, thickness_areas)
    extra_res, res = calc_score(cscan_target, post_cscan, metrics, phantom_num, thickness_areas, thickness_vals)

    if visualize:
        visualize_pred(post_cscan, a_scan_mat_pred, test_data, **vis_args)
    if provide_extra_res:
        return {**res, **extra_res}
    else:
        return res


def calc_score(c_scan_target, distance_c_scan, metrics, phantom_num, thickness_areas, thickness_vals):
    diffs = (distance_c_scan - c_scan_target)
    num_samples_thickness = []
    tot = 0
    # misses = 0
    extra_res = {}
    res = {}
    for metric in metrics:
        for i, thickness in enumerate(thickness_vals):
            key = metric + '_' + str(np.round(thickness, 3))

            ring_area = thickness_areas[i]
            # TODO - think if measure miss detections
            # pos = np.logical_and(ring_area, np.logical_not(miss_detections))
            # neg = np.logical_and(ring_area, miss_detections)
            ring_diffs = diffs[ring_area]
            num_samples_thickness.append(np.count_nonzero(ring_area))
            # miss_detections_ratio = np.count_nonzero(neg) / num_samples_thickness[i]

            if metric == 'mae':
                score = np.average(abs(ring_diffs))
            elif metric == 'rmse':
                # score = np.avg(np.sqrt(ring_diffs ** 2)) / num_samples_thickness[i]
                score = np.sum(np.sqrt(ring_diffs ** 2)) / num_samples_thickness[i]
            elif metric == 'relative_mae':
                score = np.average(abs(ring_diffs)) / thickness

            std = np.std(ring_diffs)
            bias = np.mean(ring_diffs)

            if i < len(thickness_areas) - 1:
                res[key] = [score]
                extra_res[key + '_std'] = std
                # extra_res[key + '_miss_ratio'] = miss_detections_ratio
                # bias2 = res[key]**2 - extra_res[key + '_std']**2
                extra_res[key + '_bias'] = bias
                # assert bias2 - extra_res[key + '_bias']**2 < 10e-5, 'wrong calculation of mse\\var\\bias'
                tot += res[key][0] * num_samples_thickness[i]
                # misses += extra_res[key + '_miss_ratio'] *  num_samples_thickness[i]
                tot_var = extra_res[key + '_std'] * num_samples_thickness[i]
                tot_bias = extra_res[key + '_bias'] * num_samples_thickness[i]

            # thinnest layer
            else:
                res[metric + '_thinnest_s' + str(phantom_num)] = [np.round(score, 3)]
                extra_res[key + '_std'] = std
                extra_res[key + '_bias'] = bias

            # total of all layers but thinnest one
            res[metric + '_tot_s' + str(phantom_num)] = [np.round(tot / np.sum(num_samples_thickness), 3)]
            # extra_res[key + 'miss_ratio'] = misses / np.sum(num_samples_thickness)
            extra_res['tot_std_s' + str(phantom_num)] = tot_var / np.sum(num_samples_thickness)
            extra_res['tot_bias_s'+ str(phantom_num)] = tot_bias / np.sum(num_samples_thickness)
            # bias2 = mse_tot_bias2 / np.sum(num_samples_thickness)
    return extra_res, res


def get_sample_args(phantom_num, vis_args):

    if phantom_num == 1:
        thickness_vals = np.flip(np.arange(0.1, 0.51, 0.1))
        example_x_locs = [100, 80, 65, 50, 40]
        example_y_locs = [100, 80, 65, 50, 40]
        cscan_lim = [0,0.6]
    else:
        inch_to_mm = 25.4
        # calibrate_to_first_sample = 2.54
        thickness_vals = np.flip(np.arange(0.03, 0.1, 0.01)) * inch_to_mm
        example_x_locs = [ 38*x+10 for x in range(8)]
        # 10, 60, 100, 140, 180, 220, 260, 290]
        example_y_locs = [20] * 8
        cscan_lim = [0,3]

    vis_args.update({'thickness_vals':thickness_vals, 'example_x_locs':example_x_locs, 'example_y_locs':example_y_locs, 'cscan_lim':cscan_lim})

    return thickness_vals, vis_args


def get_c_scan_pred(a_scan_mat_pred, test_data, layer_eval_type):
    pred = deepcopy(test_data)
    pred.a_scan_mat = a_scan_mat_pred

    if layer_eval_type == 'max_to_max':
        pred_c_scan = pred.get_c_scan(DispType.MAX_TO_MAX)
    elif layer_eval_type == 'first_two':
        pred_c_scan = pred.get_c_scan(DispType.FIRST_TWO)
    else:
        raise NameError('not a valid value for layer_eval_type')

    return pred_c_scan


def get_sample_areas(phantom_num):
    if phantom_num == 1:
        x = np.arange(-100, 100, 1)
        y = np.arange(-100, 100, 1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        rings_radius = np.arange(20, 101, 20)
        areas = []
        for i in range(len(rings_radius)):
            big_radius_circle = (xx ** 2 + yy ** 2) < ((rings_radius[i] - 1) ** 2)
            if i > 0:
                small_radius_circle = (xx ** 2 + yy ** 2) < ((rings_radius[i - 1] + 1) ** 2)
            else:
                small_radius_circle = np.zeros(big_radius_circle.shape, dtype=bool)
            ring_area = big_radius_circle != small_radius_circle
            areas.append(ring_area[:185, :])  # Crop according to data
    else:
        left_x = np.arange(6, 300, 38)
        top_y = [5] * 8
        right_x = np.arange(30, 300, 38)
        bottom_y = [30] * 8
        # for i in range(len(thicknesses)):
        # sample_gt[left_x[i]:right_x[i], top_y[i]:bottom_y[i]] = thicknesses[i]

        # rings_radius = np.arange(20, 101, 20)
        areas = []
        sample_area = np.zeros((35, 307), dtype=bool)
        for i in range(len(left_x)):
            # area = np.zeros(args.)
            cur_area = sample_area.copy()
            cur_area[top_y[i]:bottom_y[i], left_x[i]:right_x[i]] = True
            areas.append(cur_area)

    return areas


def estimate_translation_to_distance(cscan_gt, cur_c_scan, ring_areas):
    samples = 500
    target_samples = np.zeros(samples * len(ring_areas))
    pred_samples = np.zeros(samples * len(ring_areas))

    # Translate timestamps (nano seconds) into distance (mm)
    flat_cscan = cscan_gt.flatten()
    flat_cscan_pred = cur_c_scan.flatten()

    for i, ring_area in enumerate(ring_areas):
        ring = ring_area.flatten()
        p = np.random.permutation(np.nonzero(ring))
        target_samples[i * samples:(i + 1) * samples] = flat_cscan[p[0, :samples]]
        # pred_samples[i * samples:(i + 1) * samples] = flat_cscan_pred[p[0, :samples]]
        pred_samples[i * samples:(i + 1) * samples] = [np.median(flat_cscan_pred[p[0, :samples]])]*samples

    translation_coef = np.sum(target_samples * pred_samples) / np.sum(pred_samples ** 2)
    return translation_coef


def convert_time_to_distance(cur_c_scan, material_velocity, translate_time_to_thickness_from_data=False, cscan_gt=[], ring_areas=[], sampling_rate=100):

    #
    if translate_time_to_thickness_from_data:
        c_scan = estimate_translation_to_distance(cscan_gt, cur_c_scan, ring_areas[:2]) * cur_c_scan
    else:
        c_scan = material_velocity * cur_c_scan / (sampling_rate * 2)

    return c_scan


def convert_distance_to_time(cur_c_scan, material_velocity, translate_time_to_thickness_from_data=False, cscan_gt=[], ring_areas=[], sampling_rate=100):
    #
    if translate_time_to_thickness_from_data:
        c_scan = cscan_gt / estimate_translation_to_distance(cscan_gt, cur_c_scan, ring_areas[:4])
    else:
        c_scan = cscan_gt * (sampling_rate * 2) / material_velocity

    return c_scan


def post_process_c_scan(c_scan, phantom_num,smoothen =True):
    # Smooth and crop according to gt
    if smoothen:
        c_scan = scipy.signal.medfilt2d(c_scan, 3)
    if phantom_num == 1:
        post_c_scan = c_scan[6:, 1:201]  # crop square around circle
        # post_misses = misses[6:, 1:201]
    else:
        post_c_scan = c_scan
        # post_misses = misses

    return post_c_scan


def visualize_pred(distance_c_scan, a_scan_mat_pred, a_scan_data, thickness_vals, cscan_lim, res_path, example_x_locs, example_y_locs, save_cscan =True, save_ascan =True, timeframe_units=True, phantom_num=1):

    # show_cscan(a_scan_data.c_scan_gt, cscan_lim, example_x_locs, example_y_locs, thickness_vals, res_path, save_cscan, timeframe_units, phantom_num,draw_gt=False)
    show_cscan(distance_c_scan, cscan_lim, example_x_locs, example_y_locs, thickness_vals, res_path, save_cscan, timeframe_units, phantom_num)

    show_a_scans(a_scan_data, example_x_locs, example_y_locs, res_path, a_scan_mat_pred, save_ascan)


def show_cscan(c_scan, cscan_lim, ascan_x_locs, ascan_y_locs, thickness_vals, res_path,save_cscan,timeframe_units,phantom_num,draw_gt=False):

    cscan_lim[1] += phantom_num*timeframe_units*55
    c_scan[c_scan < cscan_lim[0]] = cscan_lim[0]
    c_scan[c_scan > cscan_lim[1]] = cscan_lim[1]
    fig, ax = plt.subplots()
    cscan = ax.imshow(c_scan, cmap='gray', interpolation='nearest')
    cscan.set_clim(cscan_lim[0], cscan_lim[1])
    ax.axis('off')

    if draw_gt:
        '''phantom1'''###
        # ascan_x_locs = [86]*5
        # ascan_y_locs = [103,73,53,33,13]
        # for i, t in enumerate(thickness_vals):
        #     if i>2:
        #         color = 'white'
        #     else:
        #         color = 'black'
        #     plt.text(ascan_x_locs[i], ascan_y_locs[i], str(np.round(t, 2))+'mm', color=color, fontsize=15)

        '''phantom2'''
        # 10, 60, 100, 140, 180, 220, 260, 290]
        ascan_y_locs = [20] * 8
        ascan_x_locs = [38*x+10 for x in range(8).__reversed__()]

        for i, t in enumerate(thickness_vals):
            if i<3:
                color = 'white'
            else:
                color = 'black'
            plt.text(ascan_x_locs[i], ascan_y_locs[i], str(np.round(t, 2))+'mm', color=color, fontsize=10)
        # plt.text([95,95,95], ascan_y_locs[10,30,50], str(np.round(t, 2))+'mm', color=color, fontsize=10)
        # plt.text([95,95], ascan_y_locs[70,100], str(np.round(t, 2)), color='white', fontsize=10)
    plt.tight_layout()

    # for i, t in enumerate(thickness_vals):
    #     plt.text(ascan_x_locs[i],ascan_y_locs[i],str(np.round(t,2)),color='g',fontsize=10)
    name = 'C-scan'
    i = 0
    while files_operations.is_file_exist(os.path.join(res_path + name + '.jpg')):
        name += str(i)
        i + 1
    if save_cscan:
        plt.savefig(os.path.join(res_path + name + '.jpg'),transparent=True,bbox_inches='tight')
    # plt.show()
    plt.close()


def show_a_scans(data, x_loc, y_loc, save_path=None, pred_sample=None, save_ascan=True,hold=False,):

    # Present A-scans from different locations
    # a_scan_locations = [105, 85, 65, 55, 40]
    ind_mat = data.wave_indx_mat
    cur_a_scans = [];
    raw_a_scans = [];
    # envelopes = [];
    plt.figure(figsize=(15, 20))
    for a in range(len(x_loc)):
        # make second echo with negative value to represent 180 degrees difference in phase
        if pred_sample is not None:
            a_scan = get_a_scan(pred_sample, ind_mat, y_loc[a], x_loc[a])

            cur_a_scans.append(a_scan)
        raw_a_scan = get_a_scan(data.a_scan_mat, ind_mat, y_loc[a], x_loc[a])
        raw_a_scan /= np.linalg.norm(raw_a_scan)
        raw_a_scans.append(raw_a_scan)
        # envelopes.append(abs(hilbert(raw_a_scans[-1])))

    num_rows = len(raw_a_scans)

    for r in range(num_rows):
        # cur_a_scans[r] = cur_a_scans[r] + r
        raw_a_scans[r] = raw_a_scans[r] + r
        # envelopes[r] = envelopes[r] + r
        # lines = [raw_a_scans[r], envelopes[r]]
        lines = [raw_a_scans[r]]
        labels = ['a_scan', 'envelope']
        if pred_sample is not None:
            cur_a_scans[r] = cur_a_scans[r] + r
            lines.append(cur_a_scans[r])
            labels.append('prediction')
        if r == 0:

            plt.plot(raw_a_scans[r], color='r', label='a_scan')
            # plt.plot(envelopes[r], color='g', label='envelope')
            if pred_sample is not None:
                plt.plot(cur_a_scans[r], color='y', label='prediction')
        else:
            plt.plot(raw_a_scans[r], color='r')
            # plt.plot(envelopes[r], color='g')
            if pred_sample is not None:
                plt.plot(cur_a_scans[r], color='y')

    plt.legend()

    if save_ascan:
        plt.savefig(save_path+'.jpg')

    plt.close()


def get_a_scan(a_scan_mat, indx_mat, i_indx, j_indx=None):

    if j_indx is None:
        return a_scan_mat[i_indx, :]
    else:
        index = indx_mat[i_indx, j_indx]
        return a_scan_mat[index, :]