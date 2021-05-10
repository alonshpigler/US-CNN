import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from progressbar import progressbar
from scipy.signal import convolve, hilbert
from sklearn.metrics import auc


def eval_simulation(y_pred_signal,
                    test_data_gt,
                    res_path,
                    rp_thresh_values = 200,
                    epsilon = 2,
                    show_eval = True,
                    max_det=40,
                    save_errors = True):

    """
    A function evaluating the restoration of spikes in signals.

    Inputs:
    ------
    y_pred_signal:
    y_true: list of N lists, each containing the ground truth detections.
    spike_min (optional): For generating average precison graph, if there is a known                minima for the spikes.
    spike_max (optional): For generating average precison graph, if there is a known                maxima for the spikes.
    spike_max (optional): For generating average precison graph, determines how many                iterations of epsilon to examine.
    ------
    """

    N, signal_length = y_pred_signal.shape

    y_true = signal2detection(test_data_gt.spikes)
    found = [0] * len(y_true)
    y_true['found'] = found

    y_pred = signal2detection(y_pred_signal, max_det=max_det)
    hits = [0] * len(y_pred)
    y_pred['hit'] = hits

    for i in progressbar(range(N)):

        image_dets = y_pred[(y_pred.image_id == i)]
        image_gts = y_true[(y_true.image_id == i)]

        for det in range(len(image_dets)):
            hit = 0
            # finding match between each pred and gt
            for spike in range(len(image_gts)):

                pulse_ind =np.where(test_data_gt.echo_inds[i,:] == image_gts.ind.values[spike])
                echo = test_data_gt.separated_echoes[i,pulse_ind,:].squeeze()
                if image_gts.iloc[[spike]].found.values[0] == 0:
                    hit = hit_or_miss(image_dets.iloc[[det]], image_gts.iloc[[spike]],echo, signal_length, epsilon, show_eval)

                # if detection hit gt
                if hit:
                    image_gts.set_value(image_gts.iloc[[spike]].index.values[0], 'found', 1)
                    y_true.set_value(image_gts.iloc[[spike]].index.values[0], 'found', 1)
                    y_pred.set_value(image_dets.iloc[[det]].index.values[0], 'hit', 1)
                    break

    F1s = []
    precision_vals = []
    recall_vals =[]
    res = {}
           # 'FAR': []}
    if len(y_pred) > 0:
        rp_thresh_values = min(rp_thresh_values, len(y_pred))
    percentiles = np.linspace(1, 0, rp_thresh_values)

    # Run on threshold to produce ROC curve
    for percentile in percentiles:

        curr_pred = y_pred[y_pred.score >= y_pred[y_pred.score > 0].quantile(percentile).score]

        hits = len(curr_pred[y_pred.hit == 1])
        # found =len(y_true[y_true.found ==1])
        false_alarms = len(curr_pred) - hits
        misses = len(y_true) - hits

        if hits + false_alarms > 0:
            precision = hits / (hits + false_alarms)
            # last_precision = precision
        else: precision = 0

        if hits + misses > 0:
            recall = hits / (hits + misses)
        else: recall = 0

        if recall or precision: F1 = 2 * precision * recall / (precision + recall)
        else: F1 = 0
        FAR = false_alarms / N

        precision_vals.append(np.round(precision, 3))
        recall_vals.append(np.round(recall, 3))
        F1s.append(np.round(F1, 3))
        # res['FAR'].append(np.round(FAR, 3))
    F1_best = np.round(max(F1s),3)

    # make_rp_monotone

    precision_vals = make_precision_monotone(precision_vals)
    res['f1']= [F1_best]
    # visualize_results()
    if len(precision_vals) > 1:
        res['auc'] = [np.round(recall_precision_curve(recall_vals, precision_vals, res_path), 3)]
        draw_errors(y_pred, y_true, res_path, save_errors = save_errors,signal_n=signal_length)
    else:  # case where zero solution is learned
        res['auc'] = 0
    return res


def make_precision_monotone(precision_vals):
    temp = []
    last = 0
    for pre in reversed(precision_vals):
        last = max(last, pre)
        temp.append(last)

    return list(reversed(temp))


def detection2signal(detections,T=500):

    image_ids = detections.image_id.unique()
    N = len(image_ids)

    signals = np.zeros([N,T])
    if len(detections) > 0:
        for i in range(N):
            image_id = image_ids[i]
            image_det = detections[detections.image_id == image_id]
            for d in range(len(image_det)):
                signals[i,image_det.ind.values[d]] = image_det.score.values[d]

    return signals


def signal2detection(output_map, min_confidence=0.01,max_det=1000):

    N, T = output_map.shape
    # detections = []
    # confidences = -np.sort(-np.abs(output_map))
    confidences = np.zeros(output_map.shape)
    for i in range(N):
        confidences[i,:] = sorted(output_map[i, :], key=abs, reverse=True)

    confidences[:,min(T-1,max_det):] =  0
    locations = np.argsort(-np.abs(output_map))

    inds = []
    scores = []
    image_ids = []
    pulse_ids = []
    count = 0
    for i in range(N):
        for detection in range(T):
            if np.abs(confidences[i,detection]) > min_confidence:
                inds.append(int(locations[i, detection]))
                scores.append(np.abs(confidences[i, detection]))
                image_ids.append(int(i))
                pulse_ids.append(int(count))
                count += 1
            else:
                break
    detections = {
                  'ind': inds,
                  'score': scores,
                  'image_id': image_ids,
                  'pulse_id': pulse_ids
                  }
    detections = pd.DataFrame(detections)
    detections.sort_values('score',ascending=False)
    return detections


def hit_or_miss(det, gt, pulse, T, epsilon, show_eval=True):

    # detection echo to pulse
    det_signal = np.zeros(T)
    det_signal[det.ind] = det.score
    det_signal = convolve(det_signal, pulse, 'same')
    det_signal_env = np.abs(hilbert(det_signal))

    # gt echo to pulse
    gt_signal = np.zeros(T)
    gt_signal[gt.ind] = gt.score
    gt_signal = convolve(gt_signal, pulse, 'same')
    gt_signal_env = np.abs(hilbert(gt_signal))

    # check if hit or miss
    hit = np.linalg.norm(det_signal_env-gt_signal_env)**2 / np.linalg.norm(gt_signal_env)**2  < epsilon
    axis = np.arange(T)

    # visualize
    if show_eval:
        draw_eval(gt,det,gt_signal, det_signal, gt_signal_env, det_signal_env, hit,axis)

    return hit


def recall_precision_curve(recall, precision, save_path, show_fig=0):
    dx = np.diff(recall)
    if np.any(dx < 0) and not np.all(dx <= 0):
        return 0
    auc_res = auc(recall,precision)
    plt.figure()
    plt.plot(recall, precision)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.ylim((0, 1.01))
    plt.xlim((0, 1.01))
    plt.title('recall precision curve with AUC=' + str(auc_res))
    plt.savefig(save_path+'recall_precision.jpg')
    if show_fig:
        plt.show()
    plt.close()
    return auc_res


def draw_eval(gt,det,gt_signal, det_signal, gt_signal_env, det_signal_env, hit,axis):

    NMSE = np.round(np.linalg.norm(det_signal_env - gt_signal_env)**2 / np.linalg.norm(gt_signal_env)**2, 2)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 7),
                           sharex=True, sharey=True)
    if hit:
        st = 'Hit!'
    else:
        st='Miss!'
    fig.suptitle(st +' gt ind = ' + str(gt.ind.values[0]) + ',det ind = ' + str(np.round(det.ind.values[0], 2)) + ', gt amp = ' + str(
        np.round(gt.score.values[0],2)) + ',det amp = ' + str(np.round(det.score.values[0], 2)))
    plt.gray()
    if hit:
       ax[0, 0].plot(axis, det_signal, color='g',label = 'detection, τ=' +str(det.ind.values[0])+',x='+str(np.round(det.score.values[0], 2)));
    else:
      ax[0, 0].plot(axis, det_signal, color='r', label='detection, τ=' +str(det.ind.values[0])+',x='+str(np.round(det.score.values[0], 2)));
    ax[0, 0].plot(axis, gt_signal, color='b',label = 'ground truth, τ=' +str(gt.ind.values[0])+',x='+str(np.round(gt.score.values[0], 2)));
    ax[0, 0].legend()
    ax[0, 0].set_title('Normalized MSE = '+ str(NMSE) + ' - ' + st)
    ax[0, 1].plot(axis, det_signal - gt_signal);
    ax[0, 1].set_title('difference between det and gt is ' + str(
        np.round(np.linalg.norm(det_signal - gt_signal)**2 / np.linalg.norm(gt_signal), 2)**2))
    ax[1, 0].plot(axis, det_signal_env, color='b');
    ax[1, 0].plot(axis, gt_signal_env, color='g');
    ax[1, 0].set_title('det (blue) and gt (green) envelopes')
    ax[1, 1].plot(axis, det_signal_env - gt_signal_env);
    ax[1, 1].set_title('difference between det and gt envelopes is ' + str(
        np.round(np.linalg.norm(det_signal_env - gt_signal_env)**2 / np.linalg.norm(gt_signal_env)**2, 2)))
    plt.show()


def draw_error(y_pred, y_true, error_to_draw='false alarm', T=500, save_errors=0, save_path='.', samples_num = 7, errors_in_fig = 7, show_fig=0, fig_name='errors', draw_only_errors=False):

    axis = np.arange(0, T*100)
    n = len(y_true.image_id.unique())

    sample = 0
    fig_num = 1
    i = 0
    while sample < samples_num | i < n - 1:
        curr_sample = 0
        fig, ax = plt.subplots(nrows=errors_in_fig, ncols=1, figsize=(10, 10),
                               sharex=True, sharey=False);
        for i in range(n):

            false_alarms_det = y_pred[(y_pred.hit == 0) & (y_pred.image_id == i)]
            missed_echos = y_true[(y_true.found == 0) & (y_true.image_id == i)]

            if draw_only_errors:
                if error_to_draw == 'false alarm':
                    condition = len(false_alarms_det)>0
                else:
                    condition = len(missed_echos)>0
            else:
                condition = True

            if condition:
                hits_det = y_pred[(y_pred.hit == 1) & (y_pred.image_id == i)]
                found_echos = y_true[(y_true.found == 1) & (y_true.image_id == i)]
                false_alarms = detection2signal(false_alarms_det,T)
                hits = detection2signal(hits_det,T)
                missed = detection2signal(missed_echos,T)
                found = detection2signal(found_echos,T)
                line_labels = ['detected gt', 'hits', 'missed_gt','false_alarms']
                l1 = ax[curr_sample].plot(np.squeeze(found), color='g')[0]
                l3 = ax[curr_sample].plot(np.squeeze(missed), color='y')[0]
                l4 = ax[curr_sample].plot(np.squeeze(false_alarms), color='r')[0]
                l2 = ax[curr_sample].plot(np.squeeze(hits), color='b')[0]

                ax[curr_sample].set_title(fig_name + error_to_draw + ' - signal number' + str(i))
                curr_sample += 1
                sample += 1

                if (curr_sample == errors_in_fig) | (i == n - 1):
                    if save_errors:
                        fig.legend([l1,l2,l3,l4], line_labels, loc='center right')
                        plt.savefig(save_path+error_to_draw +'.jpg')
                        if show_fig:
                            plt.show()
                    plt.close(fig)
                    fig_num = +1
                    break



def draw_errors(y_pred, y_true, res_dir, errors_to_draw ='misses', save_errors=True, samples_num = 7, errors_in_fig = 7, show_errors=0, fig_name='errors', signal_n=5):

    """
        Draws false alarms and miss detections. see show_error() for more details
    :param errors_to_draw:
        - 'both' to show false alarms and miss detections
        - 'false alarm' to show only false alarms
        - 'miss detections' to show only miss detections
    """

    # show false alarms
    if errors_to_draw == 'both' or  errors_to_draw =='false alarms':
        draw_error(y_pred, y_true, error_to_draw='false alarm', save_errors=save_errors, save_path=res_dir, samples_num = samples_num, errors_in_fig = errors_in_fig, show_fig=show_errors, fig_name=fig_name,T=signal_n)
    if errors_to_draw == 'both' or  errors_to_draw =='misses':
        draw_error(y_pred, y_true, error_to_draw='miss detections', save_errors = save_errors, save_path = res_dir, samples_num = samples_num, errors_in_fig = errors_in_fig, show_fig=show_errors, fig_name=fig_name,T=signal_n)

