import logging
import os
import numpy
import torch
from configuration import config
from data_layer.prepare_data import prepare_data
from data_layer.signal_analysis import get_signal_params
from data_layer.generate_simulation import generate_data
from eval_layer.post_process import post_process
from model_layer.net_pipe import fit
from utils.files_operations import is_file_exist, load_pickle, save_to_pickle, write_dict_to_csv_with_pandas, get_current_working_directory
from utils.timer import Timer
import matplotlib

matplotlib.use('TkAgg')


def main(SAVE_STATE, USE_CASHE, exp_args={}):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    args, paths, funcs = config.set_configuration(exp_args)

    exp_name = 'US-CNN with ' + args.difficulty + ' data randomization on ' + args.test_data_type
    logging.info("==== Executing exp " + exp_name + " ====")

    if args.use_ref_signal:
        signal_params = get_signal_params(paths['REF_SIGNAL'], **args.analyze_signal_kwargs)
        args = config.update_signal_params(signal_params, args)

    data = run_step(generate_data, [], args.data_kwargs, SAVE_STATE['DATA'], USE_CASHE['DATA'], paths['SIM_DATA'])

    if not args.test_on_real_data:   # Generate simulated test data
        data['test'] = run_step(generate_data, [], args.test_data_kwargs, SAVE_STATE['DATA'], USE_CASHE['DATA'], paths['TEST_DATA'])

    prepared_data = prepare_data(data, **args.prepare_data_kwargs)

    args.pred_kwargs['net_state_dict'] = run_step(fit, prepared_data['dataloaders'], args.fit_kwargs, SAVE_STATE['FIT'], USE_CASHE['FIT'], paths['MODEL'])
    pred = run_step(funcs['predict'], prepared_data['dataloaders']['test'], args.pred_kwargs, SAVE_STATE['PREDICT'], USE_CASHE['PREDICT'], paths['PREDICT'])
    pred_post_process = post_process(pred, **args.post_process_kwargs)
    res = funcs['eval'](pred_post_process, prepared_data['data']['test'], **args.eval_kwargs)

    return res


def run_step(func, input, func_kwargs, save_state, use_cache, path):
    step_name = func.__name__
    logging.info("================ Start: " + step_name + " ===========")
    if not use_cache or not is_file_exist(path):
        timer = Timer('Executing ' + step_name + '...').start()
        if len(input) > 0:  # if there is input
            output = func(input, **func_kwargs)
        else:
            output = func(**func_kwargs)
        timer.end()

        if save_state:
            timer = Timer('Saving ' + step_name + ' ...').start()
            if func.__name__ == 'fit':
                torch.save(output, path)
            else:
                save_to_pickle(output, path)
            timer.end()

    else:
        timer = Timer('Loading ' + step_name + ' from file...').start()
        if func.__name__ == 'fit':
            # output = CNN1D(1, **func_kwargs['net_args'])
            output = torch.load(path)
        else:
            output = load_pickle(path)
        timer.end()
    logging.info("================ Finished: " + step_name + " ===========")

    return output



if __name__ == '__main__':

    exp_num=1

    project_dir = get_current_working_directory()
    data_dir = os.path.join(project_dir, 'Data')

    project_output_dir = os.path.join(project_dir, 'outputs')
    res_dir = os.path.join(project_output_dir, 'results',str(exp_num))
    model_dir = os.path.join(project_output_dir,'models',str(exp_num))

    debug = False
    seed = 100
    data_randomization = 'no_rand'  # Defines level of randomization in training data. options: '{'no_rand',  'narrow', 'wide'].
    test_data_type = 'phantom1'   # defines testing data. options:  ['no_rand','narrow', 'wide', 'phantom1','phantom2']

    config_kwargs = {
        'debug': debug,
        'pre_defined_snr': False,
        'num_samples':500,
        'epochs':10
    }
    SAVE_STATE = {
        'DATA': True,
        'FIT': True,
        'PREDICT': True
    }
    USE_CASHE = {
        'DATA': True,
        'FIT': True,
        'PREDICT': True
    }
    if debug:
        for key in SAVE_STATE.keys():
            SAVE_STATE[key] = False
        for key in USE_CASHE.keys():
            USE_CASHE[key] = False


    res_final = {'method':['US_CNN']}
    if not debug:
        save_to_pickle(config_kwargs, os.path.join(res_dir, 'args.pkl'))
    exp_kwargs = {**config_kwargs, 'difficulty': data_randomization, 'test_data_type': test_data_type, 'data_dir':data_dir, 'res_dir':res_dir, 'model_dir':model_dir}

    torch.manual_seed(seed)
    numpy.random.seed(seed)

    res = main(SAVE_STATE, USE_CASHE, exp_kwargs)

    res_final = {**res_final, **res}
    print(res_final)

    write_dict_to_csv_with_pandas(res_final, os.path.join(res_dir, 'results_on_' + test_data_type + '.csv'))

