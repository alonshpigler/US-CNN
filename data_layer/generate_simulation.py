import numpy as np
import matplotlib.pyplot as plt
import progressbar
from scipy.signal import hilbert
import pickle
import torch
import math
from utils import utils


def generate_data(
  var_range=0,  # level of randomization for [frequency (db), s (microseconds) , phi , (radians) ]
  phase_range=0.1 * math.pi,
  freq_range=0,
  min_num_echoes=1,  # minimal number of echoes in a signal
  max_num_echoes=8,  # maximal number of echoes in a signal
  avg_distance_between_echoes_std=3,  # average overlap of echoes (in sigma)
  distance_between_echoes_range_std=[1.5, 6],  # lower and upper limit of overlap of echoes (sigma)
  num_samples=3000,  # number of signals in simulated data
  gt_s_ratio=0.3,
  # reference echo params
  phase=2.065,
  var=0.11,
  second_phase=None,  # phase difference of 2nd echo from 1st echo
  fc=5,
  f_low=4,
  pulse_n=1,
  fs=100,
  signal_n=5,
  amp_range=[0.1, 1],
  echo_time_arrival=[0.5, 2],
  tau_dist='exp',  # distribution of distance between echoes. options: {'uniform', 'gamma', 'exp'}
  pulse_params_dist='uniform',  # distribution of echo parameters
  is_opposite_from_first_phase=False,
  with_negative=False,  # if 1, amplitude of echoes with opposite phase is negative
  neg_amp_ratio=1,  # in the range [0,1], sets percentage of echoes generated with opposite phase (starting from second echo)
  snr=20,
  with_gaussian_gt=True,
  # test=False,
  #   test_data_type='',
  show_overlap_distribution=False,
  normalize_echos=True,
  normalize_signal=True,
  show_signal_instance=False,
):
    # if test:
    #     num_samples = 2000

    draw_tau, is_rand_freq, is_rand_var, is_rand_phase, var_dist, phase_dist, freq_dist = \
        set_randomization_params(tau_dist, avg_distance_between_echoes_std, distance_between_echoes_range_std, fc, f_low, var, phase, var_range, phase_range, freq_range, pulse_params_dist)
    data, t_arr, sep_echoes, n_echoes = data_generation_init(signal_n, fs, num_samples, max_num_echoes, min_num_echoes)

    data['target'], data['echo_params'] = create_echoes(amp_range, freq_dist, phase_dist, var_dist, draw_tau, echo_time_arrival, fc, fs, distance_between_echoes_range_std,
                                                        is_opposite_from_first_phase, n_echoes, neg_amp_ratio, normalize_echos, num_samples, phase, is_rand_freq, is_rand_phase, is_rand_var, var,
                                                        second_phase, signal_n, with_negative, show_overlap_distribution)

    signals = create_signals(data['echo_params'], fs, n_echoes, num_samples, signal_n, normalize_signal)
    data['input'] = add_noise(torch.tensor(signals).cuda(), snr)

    # generate single echo representations
    mid = pulse_n / 2
    tau = (torch.ones(len(data['echo_params']['ind'])).cuda() - mid) * fs
    separated_echoes = single_spike_to_signal(data['echo_params']['fc'], data['echo_params']['phi'], data['echo_params']['s'], tau, data['echo_params']['amp'], pulse_n, fs).cpu().numpy()

    data['separated_echoes'] = np.zeros((num_samples, max_num_echoes, pulse_n * fs))
    echo_start = 0
    for i in range(num_samples):
        for n in range(n_echoes[i]):
            # ordered_echoes = separated_echoes[echo_start,:]

            data['separated_echoes'][i, n, :] = separated_echoes[echo_start, :]
            echo_start += 1

    if with_gaussian_gt:
        data['target_gaus'] = create_signals(data['echo_params'], fs, n_echoes, num_samples, signal_n, normalize_signal, only_gaussian=True, gaussian_std_shrink_ratio=gt_s_ratio)

    if show_signal_instance:
        visF.nice_plot([data['input'][0, :], data['target'][0, :], abs(hilbert(input[0, :]))], labels=['signal', 'echo', 'envelope'])

    return data


def create_echoes(amp_range, draw_freq, draw_phi, draw_s, draw_tau, echo_time_arrival, fc, fs, gap, is_opposite_from_first_phase, n_echoes, neg_amp_ratio, normalize_echos, num_samples, phi,
                  rand_omega, rand_phi, rand_s, s, second_phi, signal_n, with_negative, show_overlap_distribution):
    overlaps = []
    spikes = np.zeros([num_samples, signal_n * fs])
    echoes = {'ind': [], 'ind_by_signal': np.zeros([num_samples, max(n_echoes)]), 'amp': [], 's': [], 'phi': [], 'fc': [], 'signal_id': []}

    for i in progressbar.progressbar(range(num_samples)):

        n_accept = 0
        while n_accept < n_echoes[i]:
            if n_accept == 0:
                echoes, first_phase_is_opposite, overlaps = add_echo(amp_range, draw_freq, draw_phi, draw_s, draw_tau, echo_time_arrival, echoes, fc, fs, gap, i, is_opposite_from_first_phase,
                                                                     neg_amp_ratio, phi,
                                                                     rand_omega, rand_phi,
                                                                     rand_s, s, second_phi, signal_n, with_negative, first_echo=1, overlaps=overlaps)
            else:
                echoes, __, overlaps = add_echo(amp_range, draw_freq, draw_phi, draw_s, draw_tau, echo_time_arrival, echoes, fc, fs, gap, i, is_opposite_from_first_phase,
                                                neg_amp_ratio, phi, rand_omega, rand_phi,
                                                rand_s, s, second_phi, signal_n, with_negative, first_echo=0, first_phase_is_opposite=first_phase_is_opposite, overlaps=overlaps)

            spikes[i, int(echoes['ind'][-1])] = echoes['amp'][-1]
            n_accept += 1

        if normalize_echos:
            if n_accept == (n_echoes[i]):
                echo_norm = np.linalg.norm(spikes[i, :])
                spikes[i, :] /= echo_norm
                echoes['amp'][len(echoes['ind']) - n_echoes[i]:len(echoes['ind'])] /= \
                    echo_norm

    if show_overlap_distribution:
        plt.hist(overlaps)
        plt.show()

    return spikes, echoes


def create_signals(echoes, fs, n_echoes, num_samples, signal_n, normalize_signal=True, only_gaussian=False, gaussian_std_shrink_ratio=1.):
    # seperated_signals = create_seperate_simulation_signals_and_pulses(, , , pulse_n, , , signal_n, fs)
    s = [gaussian_std_shrink_ratio * x for x in echoes['s']]
    seperated_signals = single_spike_to_signal(echoes['fc'], echoes['phi'], s, echoes['ind'], echoes['amp'], signal_n, fs, only_gaussian)

    signals = torch.zeros(num_samples, round(signal_n * fs)).cuda()
    # add seperated pulses into a signal
    echo_start = 0
    for i in range(num_samples):
        for n in range(n_echoes[i]):
            signals[i, :] += seperated_signals[echo_start, :]
            echo_start += 1

    if normalize_signal:
        signals = utils.normalize(signals, axis=1)

    return signals.cpu().numpy()


def set_randomization_params(tau_dist, echo_distance_avg_sigma, gap, fc, f_low, var, phase, var_range, phase_range, freq_range, pulse_params_dist):
    # Drawing Difference in Time of Flight (DTOF)
    #  torch.rand(1) * overlap_ratio[1] +overlap_ratio[0]
    # lambda_ = 1 - overlap_ratio_exp_central_gravity
    if tau_dist == 'exp':
        draw_tau = torch.distributions.exponential.Exponential(torch.tensor([1 / (echo_distance_avg_sigma)]))
    elif tau_dist == 'gamma':
        draw_tau = torch.distributions.gamma.Gamma(torch.tensor([gap[0] + 0.1]), torch.tensor([1 / (echo_distance_avg_sigma)]))
    else:
        draw_tau = torch.distributions.uniform.Uniform(gap[0], gap[1])

    # set indicators for pulse parameters to be randomized
    rand_freq = freq_range != 0
    rand_s = var_range != 0
    rand_phase = phase_range > 0

    # set distribution functions for the pulse parameters
    # TODO - support normal dist?
    if pulse_params_dist == 'normal':
        draw_s = torch.distributions.normal.Normal(var + var * var_range / 2, var * var_range / 3)
        draw_phi = torch.distributions.normal.Normal(phase, phase_range / 3)
        draw_freq = torch.distributions.normal.Normal(fc, torch.tensor((fc + f_low) / 6).float().cuda())
    elif pulse_params_dist == 'uniform':
        draw_freq = torch.distributions.uniform.Uniform(f_low, fc)
        # draw_s = torch.distributions.uniform.Uniform(s*(1-randomize_pulse[1]/2), s*(1+randomize_pulse[1]/2))
        draw_s = torch.distributions.uniform.Uniform(var, var * (1 + var_range))
        draw_phi = torch.distributions.uniform.Uniform(phase - phase_range / 2, phase + phase_range / 2)

    return draw_tau, rand_freq, rand_s, rand_phase, draw_s, draw_phi, draw_freq


def data_generation_init(T, f, N, max_num_echoes, min_num_echoes):
    # initializations
    data = {}
    t_arr = torch.arange(0, T, 1 / f).cuda()
    sep_echoes = torch.zeros((max_num_echoes * N, len(t_arr))).cuda()
    n_echoes = torch.randint(min_num_echoes, max_num_echoes + 1, (N,)).cuda()  # draw echoes number for sample i

    return data, t_arr, sep_echoes, n_echoes


def add_echo(amp_range, draw_fc, draw_phi, draw_s, draw_tau, first_echo_time_arrival, echoes, fc_0, fs, distance_between_echo_range, i, is_opposite_from_first_phase, neg_amp_ratio, phi, rand_fc,
             rand_phi, rand_s, s_0,
             second_phi, signal_n, with_negative, first_echo, first_phase_is_opposite=0, overlaps=[]):
    # draw echo params (center frequency, variance, phase, amplitude and time of flight(TOF)
    fc = rand_fc * draw_fc.sample() + (1 - rand_fc) * fc_0
    s = rand_s * draw_s.sample() + (1 - rand_s) * s_0
    tof = draw_tof(draw_tau, first_echo_time_arrival, echoes, first_echo, fs, distance_between_echo_range, i, signal_n, s, overlaps)
    amp_with_sign, first_phase_is_opposite = draw_amplitude(amp_range, first_echo, first_phase_is_opposite, is_opposite_from_first_phase, with_negative)
    phi = draw_phase(draw_phi, first_echo, first_phase_is_opposite, is_opposite_from_first_phase, neg_amp_ratio, phi, rand_phi, second_phi, with_negative)

    update_echoes_dict(echoes, amp_with_sign, fc, i, tof, phi, s)

    return echoes, first_phase_is_opposite, overlaps


def draw_phase(draw_phi, first_echo, first_phase_is_opposite, is_opposite_from_first_phase, neg_amp_ratio, phi, rand_phi, second_phi, with_negative):
    phi = rand_phi * draw_phi.sample() + (1 - rand_phi) * phi
    if first_echo:
        is_opposite = first_phase_is_opposite
    else:
        # how to handle remaining echoes w.r.t to the first
        if is_opposite_from_first_phase:
            is_opposite = not first_phase_is_opposite
        else:
            is_opposite = torch.rand(1) < 0.5
    if not with_negative and is_opposite:  # add  to phase pi in case of opposite phase
        if second_phi is None:
            phi += math.pi
        else:
            phi += second_phi  # add pre-defined value
    return phi


def draw_tof(draw_tau, echo_time_arrival, echoes, first_echo, fs, gap, i, signal_n, s, overlaps):
    if first_echo:
        t_try = (torch.rand(1) * (echo_time_arrival[1] - echo_time_arrival[0]) + echo_time_arrival[0]) * fs
    else:
        t_try = rejection_sampling(echoes, gap, draw_tau, signal_n, fs, i, s, echoes['s'][-1], overlaps)
    # ind = extraF.round(t_try, 2)

    return t_try


def rejection_sampling(echoes, echo_tof_range, draw_tau, signal_n, fs, signal_id, s_cur, s_prev, overlaps):
    s_avg = (s_cur + s_prev) / 2
    min_distance = echo_tof_range[0] * s_avg
    max_distance = echo_tof_range[1] * s_avg
    # signal_start_boundary = (min_distance + max_distance) / 2
    signal_start_boundary = s_avg * 2
    signal_end_boundary = signal_n  - signal_start_boundary

    # signal_count = np.unique(echoes['signal_id'], return_counts=True)
    signal_inds = echoes['ind_by_signal'][signal_id, :] / fs
    nonzero_inds = np.nonzero(signal_inds)
    nonzero_elements = signal_inds[nonzero_inds]
    last_echo_tof = torch.tensor(max(nonzero_elements)).cuda()
    first_echo_tof = torch.tensor(min(nonzero_elements)).cuda()

    reject = 1
    while reject:
        reject = 0

        # overlap_ratio = max(0, (1 - draw_tau.sample())) * (echo_tof_range[1] - echo_tof_range[0]) + echo_tof_range[0]
        distance = draw_tau.sample() * s_avg # from sigma measurement to timeframes
        # distance = 3 / 2 * overlap_ratio * (s_cur + s_prev)
        if max_distance > distance > min_distance:
            if last_echo_tof + distance < signal_end_boundary:
                # draw following echo relatively to previous
                # t_try = min(last_echo_tof + draw_tau.sample().cuda(), signal_n - 1 / fs)
                t_try = min(last_echo_tof + distance.cuda(), signal_n - 1 / fs)
                # q = echoes['ind'][-1]
                # if (abs(t_try - last_echo_tof) < min_distance) or (abs(t_try - last_echo_tof) > max_distance):
                # reject = 1
            elif first_echo_tof - distance > signal_start_boundary:
                # t_try = max(0, first_echo_tof - draw_tau.sample().cuda())
                t_try = max(0, first_echo_tof - distance.cuda())
                # if (abs(t_try - last_echo_tof) < min_distance) or (abs(t_try - last_echo_tof) > max_distance):
                # reject = 1
            else:
                raise RuntimeError('Not enough space for echoes in signal')
        else:
            reject = 1
    overlaps.append(distance)

    return t_try *fs


def draw_amplitude(amp_range, first_echo, first_phase_is_opposite, is_opposite_from_first_phase, with_negative):
    amp = torch.rand(1) * (amp_range[1] - amp_range[0]) + amp_range[0]
    if first_echo:
        first_phase_is_opposite = (torch.rand(1) < 0.5).float()
        amp_with_sign = (first_phase_is_opposite) * (-with_negative * amp + (1 - with_negative) * amp) + (1 - first_phase_is_opposite) * amp
    else:
        if is_opposite_from_first_phase:
            amp_with_sign = first_phase_is_opposite * amp + (1 - first_phase_is_opposite) * (-with_negative * amp + (1 - with_negative) * amp)
        else:
            is_opposite = (torch.rand(1) < 0.5).float()
            amp_with_sign = (is_opposite) * (-with_negative * amp + (1 - with_negative) * amp) + (1 - is_opposite) * amp
    return amp_with_sign, first_phase_is_opposite


def update_echoes_dict(echoes, amp_with_sign, fc, i, ind, phi, s):
    echoes['phi'].append(phi)
    echoes['ind'].append(torch.round(ind))

    if len(echoes['signal_id']) == 0 or len(echoes['signal_id']) == i:
        echoes['signal_id'].append(i)
        echoes['ind_by_signal'][i, 0] = torch.round(ind).cpu().numpy()
    else:
        last_val = np.nonzero(echoes['ind_by_signal'][i, :])[0][-1]
        echoes['ind_by_signal'][i, last_val + 1] = torch.round(ind).cpu().numpy()

    # echoes['signal_id'].append(i)
    echoes['amp'].append(amp_with_sign)
    echoes['fc'].append(fc)
    echoes['s'].append(s)


def single_spike_to_signal(fc, phi, s, tau, amp, signal_n=5, fs=100, only_gaussian=False):
    """
    Create toch tensor of signals size NxT from pulse and echo parameters.
     See "create_seperate_simulation_signals_and_pulses" for details on parameters
     :param t_arr: Timeframs of signal (e.g. for T=5 and f=100, t_arr=0.01,...,5), repeated N_echo times. size
     :param s: density of pulse, vector of length N_echoes - the number of echoes in the data
    :param tau: signal time of flight, vector of length N_echoes - the number of echoes in the data
    :param phi: shifting parameter of cos signal, vector of length N_echoes - the number of echoes in the data
    :param fc: cycle time of pulse, vector of length N_echoes - the number of echoes in the data
    :param signal_n: Total time
    :param f: frequency of sampling
    """
    num_signals = len(amp)
    N = int(signal_n * fs)
    t_arr = torch.arange(0, N / fs, 1 / fs).float().cuda().reshape(1, N).repeat([num_signals, 1])
    fc_arr, phi_arr, s_arr, tau_arr, amp_arr = reshape_params_for_signals_gen(fc, phi, s, tau, amp, num_signals, N, fs)

    gaussian = torch.exp(-1 / 2 * (t_arr - tau_arr).pow(2) / torch.pow(s_arr, 2))

    if only_gaussian:
        signals = amp_arr * gaussian
    else:
        harmonic_func = torch.cos(2 * math.pi * fc_arr * (t_arr - tau_arr) + phi_arr)
        signals = amp_arr * gaussian * harmonic_func

    return signals


def reshape_params_for_signals_gen(omega, phi, s, tau, amp, N, T, fs):
    """
    See create_signals_from_pulse_and_echo_params.
    """

    omega_arr = torch.cuda.FloatTensor(omega).reshape(N, 1).repeat([1, T])
    phi_arr = torch.cuda.FloatTensor(phi).reshape(N, 1).repeat([1, T])
    s_arr = torch.cuda.FloatTensor(s).reshape(N, 1).repeat([1, T])
    tau_arr = torch.cuda.FloatTensor(tau).reshape(N, 1).repeat([1, T]) / fs
    amp_arr = torch.cuda.FloatTensor(amp).reshape(N, 1).repeat([1, T])

    return omega_arr, phi_arr, s_arr, tau_arr, amp_arr


def add_noise(x, snr_signal, normalize=False, randomize=False, rand_param=3):
    '''
    Adding noise to signals.
    :param x: signals, array size NxT
    :param snr_signal: signal-to-noise ratio of signal to be added
    :return: noisy signals
    '''

    noise_sigma = 10 ** (-0.05 * snr_signal)

    if len(x.shape) < 2:
        x = x.reshape(1, x.shape[0])

    x_norm = x.norm(dim=1, keepdim=True).repeat_interleave(x.shape[1], 1).cuda()
    if normalize:
        n_x = x / x_norm
    else:
        n_x = x

    signal_noise = n_x.std(dim=1, keepdim=True).repeat_interleave(x.shape[1], 1).cuda()

    if randomize:
        noise_sigma = noise_sigma.repeat_interleave(x.shape[1], 1).cuda() - rand_param +torch.randn(x.shape[1])*rand_param*2
    x_noisy = n_x + signal_noise * x.data.new(x.size()).normal_(0, noise_sigma)
    if normalize:
        x_noisy /= x_noisy.norm(dim=1, keepdim=True).repeat_interleave(x.shape[1], 1).cuda()

    return x_noisy.cpu().numpy()


def estimate_noise_old(ref_signal_path, noisy_range):
    with open(ref_signal_path, 'rb') as f:
        ref_signal = pickle.load(f)

    noisy_part = ref_signal[noisy_range[0]:noisy_range[1]]
    length_signal_noise_ratio = len(ref_signal) / len(noisy_part)

    noisy_norm = np.linalg.norm(noisy_part) * length_signal_noise_ratio
    signal_norm = np.linalg.norm(ref_signal)

    est_snr = 20 * (np.log10(signal_norm) - np.log10(noisy_norm))

    return est_snr

