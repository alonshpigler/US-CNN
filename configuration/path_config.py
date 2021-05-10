import os
from utils import files_operations


def set_paths(args, exp_params):
    paths = set_data_paths(args, exp_params)
    paths.update(set_model_paths(paths, args, exp_params))

    return paths


def set_data_paths(args, exp_params):
    paths = {}

    real_signals_dir = os.path.join(exp_params['data_dir'], 'RealPhantoms')

    '===== REFERENCE SIGNAL PATHS ====='
    if args.ref_sample == 1:
        ref_signal_name = 'phantom1_5MHz_Focus_N01_Glue_Interface'
    else:
        ref_signal_name = 'phantom2_5MHz_F1inc_D0p25_sn_D5628_pn13_0504_s'
    paths['REF_SIGNAL'] = os.path.join(real_signals_dir, 'ref_signals', ref_signal_name + '.pkl')
    paths['REF_SIGNAL_PARAMS'] = paths['REF_SIGNAL'][:-4] + '_params.pkl'


    '===== SIMULATED DATA PATHS ====='
    with_gaussian_gt = exp_params.get('with_gaussian_gt', True),
    if with_gaussian_gt:
        gt_s = exp_params.get('gt_s_ratio', 0.2)
        train_data_filename = exp_params.get('train_data_filename', args.difficulty) + '_gt_s_' + str(gt_s)
    if args.pre_defined_snr:
        train_data_filename += '_snr_' + str(args.snr)

    paths['SIM_DATA'] = os.path.join(exp_params['data_dir'], 'TrainData', train_data_filename + '_train.pkl')


    '===== TEST DATA PATHS ====='

    '===== PHYSICAL DATA PATHS ====='
    if args.phantom_num == 1 or args.phantom_num == 2:
        if args.phantom_num == 1:
            test_data_filename = 'phantom1_5MHz_Focus_N01_Glue_Interface'
        else:
            test_data_filename = 'phantom2_5MHz_F1inc_D0p25_sn_D5628_pn13_0504_s'
        paths['TEST_DATA'] = os.path.join(real_signals_dir, test_data_filename + '.hdf')
        paths['TEST_GT'] = os.path.join(real_signals_dir, args.test_data_type + '_mask.pkl')

    else:
        '=== SIMULATED TEST DATA PATHS ==='
        paths['TEST_GT'] = ''
        test_path = os.path.join(exp_params['data_dir'], 'TestData', 'Simulated')
        test_data_filename = exp_params.get('test_data_filename', args.test_data_type)
        if args.pre_defined_snr:
            test_data_filename += '_snr_' + str(args.snr)
        if not args.debug:
            paths['TEST_DATA'] = os.path.join(test_path, test_data_filename + '_test.pkl')
        else:
            paths['TEST_DATA'] = os.path.join(test_path, test_data_filename + '_mini_test.pkl')

    return paths


def set_model_paths(paths, args, exp_params):
    """"""
    '===== MODEL PATH ====='
    files_operations.make_folder(exp_params['model_dir'])
    gt_s = exp_params.get('gt_s_ratio', 0.2)
    model_filename = exp_params.get('model_filename', args.difficulty) + '_gt_s_' + str(gt_s)
    if args.pre_defined_snr:
        model_filename += '_snr_' + str(args.snr)
    paths['MODEL'] = os.path.join(exp_params['model_dir'], model_filename + '.pth')

        # paths['MODEL_TEMP'] = os.path.join(env('ModelsPath'), 'UltraSound1D', 'exp' + str(exp_num), model_type + 'temp')
    '===== PREDICTION PATH ====='
    files_operations.make_folder(exp_params['res_dir'])
    # save_dir = os.path.join(env('ExpResultsPath'), 'UltraSound1D', 'exp_' + str(exp_num),exp_name)
    pred_filename = exp_params.get('pred_filename', 'US_CNN_' + args.difficulty + '_on_' + args.test_data_type)

    paths['PREDICT'] = os.path.join(exp_params['res_dir'], 'preds', pred_filename + '_pred.pkl')
    paths['SAVE_RES_DIR'] = os.path.join(exp_params['res_dir'], args.test_data_type)

    files_operations.make_folder(paths['SAVE_RES_DIR'])
    paths['SAVE_VIS'] = os.path.join(paths['SAVE_RES_DIR'], pred_filename)

    return paths
