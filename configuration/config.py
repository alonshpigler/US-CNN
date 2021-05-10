import argparse
import math
from configuration.path_config import  set_paths
from data_layer.signal_analysis import estimate_frequency
from model_layer import net_pipe
from eval_layer.evaluate_sample import eval_sample
from eval_layer.evaluate_simulation import eval_simulation
from utils import files_operations


def set_configuration(
  exp_params={}
):
    args = get_default_args(exp_params)
    funcs = get_step_funcs(args.test_on_real_data)
    paths = set_paths(args, exp_params)

    if args.use_ref_signal:
        args.analyze_signal_kwargs = {
            'freq_range': args.freq_range,
            'noise_range': [250, 500] if args.ref_sample == 2 else [50, 100]
        }


    # parameters controlling the simulated data
    args.data_kwargs = {
        'avg_distance_between_echoes_std': exp_params.get('avg_distance_between_echoes_std', 3),
        'distance_between_echoes_range_std': exp_params.get('distance_between_echoes_range_std', [1.5, 6]),
        'with_gaussian_gt': exp_params.get('with_gaussian_gt', True),
        'gt_s_ratio': exp_params.get('gt_s_ratio', 0.2),
        'num_samples': 400 if args.debug else exp_params.get('num_samples', 10000),
        'var_range': args.var_range,
        'phase_range': args.phase_range,
        'freq_range': args.freq_range,
        'max_num_echoes': exp_params.get('max_num_echoes', 8),
        'min_num_echoes': exp_params.get('min_num_echoes', 1),
        'signal_n':exp_params.get('signal_n', 5)
    }
    if not args.test_on_real_data:
        test_freq_range, test_var_range, test_phase_range = get_data_randomization_params(args.test_data_type)
        ref_signal = files_operations.load_pickle(paths['REF_SIGNAL'])
        fc, f_low = estimate_frequency(100, ref_signal, test_freq_range)

        args.test_data_kwargs = {
            'avg_distance_between_echoes_std': exp_params.get('avg_distance_between_echoes_std', 3),
            'distance_between_echoes_range_std': exp_params.get('distance_between_echoes_range_std', [1.5, 6]),
            'f_low': f_low,
            'num_samples': exp_params.get('num_samples', 2000),
            'var_range': test_var_range,
            'phase_range': test_phase_range,
            'freq_range': test_freq_range,
            'max_num_echoes': exp_params.get('max_num_echoes', 8),
            'min_num_echoes': exp_params.get('min_num_echoes', 1),
            'signal_n':exp_params.get('signal_n', 5)
        }
    args.prepare_data_kwargs = {
        'with_gaussian_gt': exp_params.get('with_gaussian_gt', True),
        'test_on_real_data': args.test_on_real_data,
        'test_path': paths['TEST_DATA'],
        'test_gt_path': paths['TEST_GT'],
        'phantom_num': args.phantom_num
    }


    args.fit_kwargs = {
        'epochs': 45 if args.debug else exp_params.get('epochs', 80),
        'net_args': {
            'depth': exp_params.get('depth', 3),
            'big_filter_kernel': (exp_params.get('big_filter_length', 69), 1),
            'channels_out': exp_params.get('channels_out', 30),
            'dilation': exp_params.get('dilation', 2),
            'batch_norm': exp_params.get('batch_norm', False),
            'kernel_size': exp_params.get('kernel_size', 5),
        },
        'res_path': paths['SAVE_RES_DIR'],
    }

    args.pred_kwargs = args.fit_kwargs.copy()
    del args.pred_kwargs['res_path']
    del args.pred_kwargs['epochs']


    nms_range_default = 14
    if args.test_on_real_data:
        trim_threshold_default = 0.15
    else:
        trim_threshold_default = 0.02
    trim_with_constraint_default = True if args.test_on_real_data else False

    args.post_process_kwargs = {
        'nms_range': exp_params.get('nms_range', nms_range_default),
        'trim_threshold': exp_params.get('trim_threshold', trim_threshold_default),
        'max_det': 5 if args.test_on_real_data else 20,
        'trim_with_constraint': exp_params.get('trim_with_constraint', trim_with_constraint_default),
        'normalize_signal': exp_params.get('normalize_signal', True)
    }

    aluminum_material_velocity = 6.2
    ultem_material_velocity = 2.09

    if args.test_on_real_data:
        # params for evaluation on real data
        layer_eval_type_default = 'first_two' if args.phantom_num == 2 else 'max_to_max'
        args.eval_kwargs = {
            'phantom_num': args.phantom_num,
            'material_velocity': ultem_material_velocity if args.phantom_num == 1 else aluminum_material_velocity,
            'layer_eval_type': exp_params.get('layer_eval_type', layer_eval_type_default),
            'translate_time_to_thickness_from_data': exp_params.get('translate_time_to_thickness_from_data', False),
            'timeframe_units': exp_params.get('timeframe_units', False),
            'metrics': exp_params.get('metrics', ['mae']),
            'vis_args': {
                'res_path': paths['SAVE_VIS'],
                'timeframe_units': exp_params.get('timeframe_units', False),
                'phantom_num': args.phantom_num
            }
        }
    else:
        # params for evaluation on simulated data
        args.eval_kwargs = {
            'epsilon': exp_params.get('epsilon', 0.1),
            'show_eval': exp_params.get('show_eval', False),
            'rp_thresh_values': exp_params.get('rp_thresh_values', 50),
            'max_det': exp_params.get('max_det', 40),
            'res_path': paths['SAVE_VIS']
        }
    return args, paths, funcs


def get_main_exp_params(args, exp_params):
    # get main experiment parameters
    args.test_data_type = exp_params.get('test_data_type', 'narrow')
    args.difficulty = exp_params.get('difficulty', 'narrow')
    check_input(args.test_data_type, args.difficulty)
    return args


def get_default_args(exp_params={}):

    parser = argparse.ArgumentParser(description='experiment parameters')
    args = parser.parse_args()

    args = get_main_exp_params(args, exp_params)
    args.pre_defined_snr = exp_params.get('pre_defined_snr', False)
    args.snr = exp_params.get('snr', 20)

    # get default
    args.freq_range, args.var_range, args.phase_range = get_data_randomization_params(args.difficulty)
    args.debug = exp_params.get('debug', False)

    args.use_ref_signal = exp_params.get('use_ref_signal', True)
    args.ref_sample = exp_params.get('ref_sample', 2)

    if args.test_data_type == 'phantom1':
        args.phantom_num = 1
    elif args.test_data_type == 'phantom2':
        args.phantom_num = 2
    else:
        args.phantom_num = 0

    args.test_on_real_data = args.phantom_num != 0

    return args


def get_step_funcs(test_on_real_data):

    if test_on_real_data:
        eval_func = eval_sample
    else:
        eval_func = eval_simulation

    funcs = {
        'predict': net_pipe.test,
        'eval': eval_func
    }

    return funcs


def check_input(data_type, difficulty):
    data_type_options = ['phantom1', 'phantom2', 'no_rand', 'narrowest', 'narrow', 'wide', 'widest']
    if data_type not in data_type_options:
        raise NameError('Data type not supported')
    difficulty_options = [ 'no_rand', 'narrowest', 'narrow', 'wide', 'widest']
    if difficulty not in difficulty_options:
        raise NameError('Data difficulty type not supported. ')


def get_data_randomization_params(data_type):

    if data_type == 'no_rand':
        freq_range = 0  # in db
        var_range = 0  # in ratio
        phase_range = 2 * math.pi  # in radians

    if data_type == 'narrowest':
        freq_range = -1.5  # in db
        var_range = 0.1  # in ratio
        phase_range = 2 * math.pi  # in radians

    elif data_type == 'narrow':
        freq_range = -3  # in db
        var_range = 0.2  # in ratio
        phase_range = 2 * math.pi  # in radians

    elif data_type == 'wide':
        freq_range = -4.5  # in db
        var_range = 0.3  # in ratio
        phase_range = 2 * math.pi  # in radians

    elif data_type == 'widest':
        freq_range = -6  # in db
        var_range = 0.4  # in ratio
        phase_range = 2 * math.pi  # in radians

    return freq_range, var_range, phase_range


def update_signal_params(signal_params, args):

    args.data_kwargs.update(signal_params)
    if args.pre_defined_snr:
        args.data_kwargs['snr'] = args.snr
    if not args.test_on_real_data:
        args.test_data_kwargs.update(signal_params)
        if args.pre_defined_snr:
            args.test_data_kwargs['snr'] = args.snr

    return args
