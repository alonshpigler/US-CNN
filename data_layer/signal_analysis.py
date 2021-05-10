import math
import pickle
import numpy as np
from scipy.fftpack import hilbert
import data_layer.generate_simulation as gen_data
from utils import files_operations


def get_signal_params(ref_signal_path,
                      fs=100,
                      show_pulse=False,
                      measure_second_phase=False,
                      freq_range=-3,
                      noise_range=[],
                      snr_0=20,
                      verbose=True):

    # f_low, fc, phase, ref_signal, second_phase, snr, tau, var = estimate_signal_params(ref_signal_path, freq_desired_range, fs, measure_second_phase, noise_range, snr)

    ref_signal = files_operations.load_pickle(ref_signal_path)
    fc, f_low = estimate_frequency(fs, ref_signal, freq_range)
    envelop, max_amp, tau = estimate_tof(fs, ref_signal)
    var = estimate_std(fs, envelop, max_amp, tau)
    phase = estimate_phase_old(fc, fs, hilbert(ref_signal), var, np.arange(0, len(ref_signal)) / fs, tau)

    if measure_second_phase:
        max_trimmed_ref_signal = ref_signal.copy()
        max_trimmed_ref_signal[int(fs * (tau - 3 * var)):int(fs * (tau + 3 * var))] = 0
        second_phase = estimate_phase(max_trimmed_ref_signal, fc)
    else:
        second_phase = None
    if len(noise_range) > 0:
        snr = estimate_snr(ref_signal, noise_range)
    else:
        snr= snr_0

    signal_params = {
        'fc': fc,
        'f_low':f_low,
        'var':var,
        'phase':phase,
        'second_phase':second_phase,
        'snr':snr
    }
    if verbose:
        print('reference signal parameters: ')
        print(signal_params)

    if show_pulse:
        pulse = gen_data.single_spike_to_signal([fc], [phase], [var], [tau], [1], fs=fs, signal_n=len(ref_signal) / fs)
        normalized_pulse = pulse / pulse.norm()
        visF.nice_plot([ref_signal, normalized_pulse[0, :].cpu().numpy()], labels=['reference pulse', 'simulated pulse'])

    return signal_params


def calc_signal_frequency(input, spectrum_len=None):
    """
    Compute center cycle time T=(1/frequency)  of signal
    :param input: signal vector
    :param spectrum_len: length in the frequency domain. The bigger it is, the cycle time will be measured with higher resolution.
    :return: cycle time T
    """
    signal_len = len(input)
    if spectrum_len is None:
        spectrum_len = signal_len

    spectrum = np.arange(0, spectrum_len) / spectrum_len
    freqs = np.fft.fft(np.array(input), spectrum_len)[0:int(spectrum_len / 2)]
    fc = spectrum[np.argmax(freqs)]

    return fc


def estimate_frequency(fs, signal, bandwith_db_for_low_freq):

    # estimate the center frequency
    fft_len = 2 ** 14
    signal_fft = np.fft.fft(signal, fft_len)
    f_center = np.argmax(np.abs(signal_fft[0:int(fft_len / 2)])) * fs / fft_len

    # lower limit of frequency calculated -6db from center frequency
    # db = -3
    a2 = 10 ** (-bandwith_db_for_low_freq/20)
    fc_val = max(np.abs(signal_fft[0:int(fft_len / 2)]))
    f_low = np.where(np.abs(signal_fft) >= fc_val / a2)[0][0] * fs / fft_len

    return f_center, f_low


def estimate_tof(fs, signal):
    # estimate tau (time-of-flight) and maximum amplitude
    sig_hilbert = hilbert(signal)
    envelop = np.abs(sig_hilbert)
    dt = 1 / fs
    tau_indx = np.argmax(envelop)
    tau = tau_indx * dt
    max_amp = envelop[tau_indx]
    return envelop, max_amp, tau


def estimate_std(fs, envelop, max_amp, tau):
    # estimate sigma according to decline of amplitude to 1/5 of max-amplitude

    dt = 1/fs
    envelop_inds = np.where(envelop > max_amp / 6)[0]
    t2_min = (envelop_inds[0] * dt - tau) ** 2
    envelop_inds_diffs = np.diff(envelop_inds)
    more_than_one_echo = (envelop_inds_diffs!=1).any()
    if more_than_one_echo:
        # take the first gaussian tail
        t2_max = (envelop_inds[:-1][(envelop_inds_diffs>5)][0] * dt - tau) ** 2
    else:
        t2_max = (envelop_inds[-1] * dt - tau) ** 2
    sigma2 = -(t2_min + t2_max) / (4 * np.log(1 / 6))
    sigma = np.sqrt(sigma2)
    return sigma


def estimate_phase_old(f_center, fs, sig_hilbert, sigma, t_arr, tau):
    # estimate the phase according to phase of analytical signal
    indx_min = int((tau - sigma) * fs)
    indx_max = int((tau + sigma) * fs)
    rng = range(indx_min, indx_max)
    instant_phase = np.unwrap(np.angle(sig_hilbert[rng]))
    phase = np.average(instant_phase - 2 * np.pi * f_center * (t_arr[rng] - tau))
    phase = phase % (np.pi * 2)

    return phase


def estimate_phase(signal, f_center, tau=None, fs=100):

    dt = 1/fs
    signal_len = signal.shape[0]
    t_arr = np.arange(0, signal_len * dt, dt)

    if tau==None:
        tau = np.argmax(np.abs(hilbert(signal))) * dt

    T = 1/f_center
    indx_min = int((tau - T/2) * fs)
    indx_max = int((tau + T/2) * fs)
    rng = range(indx_min, indx_max)
    trim_t_arr = t_arr[rng]
    trim_signal = signal[rng]

    x = np.arange(0,len(trim_signal)) * dt
    sin_proj = np.sin(2*math.pi*x/T) * trim_signal
    cos_proj = np.cos(2 * math.pi * x/T) * trim_signal

    sin_phi = np.sum(sin_proj)
    cos_phi = np.sum(cos_proj)

    P = np.sqrt(sin_phi**2+cos_phi**2)
    # P /= np.linalg.norm(P)
    sin_phi /= P
    cos_phi /= P

    phase = np.arctan2(sin_phi, cos_phi) % (2 * math.pi)

    return phase


def estimate_snr(ref_signal, noise_range):
    only_noise = ref_signal[noise_range[0]:noise_range[1]]
    only_noise = np.asanyarray(only_noise)
    m = abs(only_noise.mean())
    sd = only_noise.std()
    signal_to_noise = np.where(sd == 0, 0, m / sd)
    snr_db = 20 *np.log10(1/signal_to_noise)
    return snr_db