import torch
import numpy as np


def normalize(signal, normalizer=[], action='norm', axis=0):
    """
    Normalize signal axis by norm/max of reference input signal.
    If there is no reference signal to normalize by, the signal is normalized to 1.
        :param signal:
        :param normalizer:
        :param action:
        :param axis:
        :return:
    """

    # normalize so that norm values will be 1 (if no Normalizer) or the same as the Normalizer
    if torch.is_tensor(signal) is False:
        signal = torch.tensor(signal)
    replicate = list(signal.shape)
    replicate[axis] = 1
    norm_mat = torch.reshape(signal.norm(dim=axis), replicate)
    # norm_mat = np.reshape(np.linalg.norm(signal, axis=axis), (replicate))

    if action == 'norm':
        if len(normalizer) == 0:
            norm_signal = signal / norm_mat
        else:
            if not torch.is_tensor(normalizer):
                normalizer=torch.tensor(normalizer)
            normalizer_norm_mat = torch.reshape(normalizer.norm(dim=axis), replicate)
            norm_signal = signal * (normalizer_norm_mat / norm_mat)

    # normalize so that maximal values will be 1 (if no Normalizer) or the same as the Normalizer
    if action == 'max':
        # norm_signal = signal / norm_mat
        if len(normalizer) == 0:
            norm_signal = signal / norm_signal.max()
        else:
            norm_signal = signal * (normalizer.max() / norm_signal.max())

    return norm_signal


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def round(x,n_digits=0):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float().cuda()
    rounded = (x * 10 ** n_digits).round() / (10 ** n_digits)
    return rounded


def mean_rand_variable(input):
    density = input / input.sum()
    return torch.sum(torch.tensor([i * density[i] for i in range(len(input))]))


def var_rand_variable(input):
    density = input / input.sum()
    mean = mean_rand_variable(input)
    return torch.sum(torch.tensor([(i-mean)**2 * density[i] for i in range(len(input))]))

