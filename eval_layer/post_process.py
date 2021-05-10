import numpy as np
import torch
from matplotlib import pyplot as plt
from  utils.utils import  normalize


def post_process(pred, nms_range=20, trim_threshold=0.01, show_post_stats=False, max_det=10, normalize_signal=True, trim_with_constraint = True, show_post_processing_sample = False):
    '''
    transform  output into signal with only spikes.

    Parameters:
    ----------
    :param normalize_signal:
    :param pred: prediction of wiener filter size NxT
    :param trim_threshold: each value lower than this scalar is set to 0
    :param nms_range: scalar determining range of zeroes around maximum in nms
    :param fs: frequency measurement
    :param show_post_processing: if 1, show example of wiener deconvolution after post-processing

    Output:
    -------
    pred_post_nms: prediction of wiener filter after trimming small values and performing non maxima supression, array Size NxT
    '''
    if len(pred.shape)==1:
        pred = pred.reshape(1,pred.shape[0])

    if show_post_stats:
        show_histogram(pred, 'before post processing')

    pred = NMS(pred, nms_range, max_det)


    if show_post_stats:
        show_histogram(pred, 'After NMS')

    # trim values below a threshold
    pred = normalize(pred, axis=1).cpu().numpy()
    if trim_threshold > 0:
        if trim_with_constraint:
            pred = trim_and_keep(pred, trim_threshold)
        else:
            pred = np.where(abs(pred) > trim_threshold, pred, 0)

    if show_post_stats:
        show_histogram(pred, 'After Trimming')

    if normalize_signal:
        pred = normalize(pred, axis=1).cpu().numpy()

    if show_post_processing_sample:
        plt.plot(pred[0, :])
        plt.title('signal after NMS')
        plt.show()

    return pred


def show_histogram(pred,title='histogram'):
    num_echoes_in_signal = np.count_nonzero(pred,axis=1)
    histogram, bins = np.histogram(num_echoes_in_signal)
    # plt.plot(pred[0, :])
    bins = np.round(bins, 0)[:-1]
    plt.bar(bins, histogram)
    plt.title(title)
    plt.show()


def trim_and_keep(pred, trim_threshold=0.01, min_echoes=2):
    trim_pred = np.zeros(pred.shape)
    N = len(pred)

    for i in range(N):
        signal = pred[i, :]
        non_zero = 0
        curr_trim = trim_threshold
        if np.count_nonzero(signal)>=min_echoes:
            while non_zero < min_echoes:
                trim_signal = np.where(abs(signal) > curr_trim, signal, 0)
                non_zero = np.count_nonzero(trim_signal)
                curr_trim /= 1.2
        else:
            trim_signal = signal
        trim_pred[i,:] = trim_signal

    return trim_pred


def NMS(pred, nms_range=20, max_det=10):
    """
    Apply non maximal suppression (NMS) to signal data.

    Inputs:
    ------
    :param pred: The predictions of all data, size Nx(Txf)
    :param nms_range: the range of values set to zero around a detection
    :param f: the frequency of sampling
    :param show_post_processing: if 1, the nms procedure of an example is visualized

    Output:
    -------
    :return: predictions after NMS
    """

    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    pred_abs = abs(pred)
    n, T = pred.shape
    sides = nms_range
    # pred_final = np.zeros(pred.shape)
    pred_post_nms = np.zeros(pred.shape)
    mat_inds = np.tile(np.arange(T), (n, 1))
    i = 0
    while np.any(pred != 0) and max_det >= i:
        i = i + 1
        nms_temp = np.zeros(pred.shape)  # add maximal value to final outcome
        pred_abs[np.max(pred_abs, axis=1) == 0, T - 1] = float('inf')  # marking empty instances
        max_values = np.max(pred_abs, axis=1)  # get spikes for iteration i
        max_mat = np.repeat(np.reshape(max_values, (n, 1)), T, 1)
        check_mat = (pred != float('inf')) & (pred_abs == max_mat)  # finding spikes entries
        # (without empty instances
        max_values = max_values[max_values != float('inf')]  # remove 'inf' from max_values

        # check that no more then one spike is chosen
        if not np.count_nonzero(check_mat) == len(max_values):

            a = np.sum(check_mat, axis=1)
            b = (-a).argsort()
            j = 0
            while (j < n) and (a[b[j]] > 1):
                check_mat[b[j], :] = False

                max_ind = np.argmax(pred_abs[b[j], :])
                check_mat[b[j], max_ind] = True
                j += 1

        # assigning max_values to final outcome
        nms_temp[check_mat] = pred[check_mat]
        pred_post_nms += nms_temp

        # trim edges around the maximal value to avoid multiplications
        max_inds = pred_abs.argmax(axis=1)
        middle = np.tile(max_inds, (T, 1)).T
        left = np.tile(max_inds, (T, 1)).T - sides
        left[left < 0] = 0
        right = np.tile(max_inds, (T, 1)).T + sides
        right[right >= T] = T - 1
        turn_to_zero = (mat_inds >= left) & (mat_inds <= right)
        pred_abs[turn_to_zero] = 0

    # pred_final = pred_final + pred_post_nms
    return pred_post_nms
