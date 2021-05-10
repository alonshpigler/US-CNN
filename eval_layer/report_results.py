import csv
import json
import os

import pandas as pd


def report(results,args,kwrgs, res_dir):
    save_results(results,args,kwrgs, res_dir)
    # visualize_results(results,args)


def get_keys(dict):
    keys=[]
    for key in dict:
        keys.append(key)
    return keys


def save_results(results,args,kwargs,res_dir):
    """
    Save results and experiment arguments to csv
    :param args: NameSpace including experiment arguments.
    :param results: Dictionary containing experiment results
    :return: ---
    """
    # set variable to save
    res_path = os.path.join(res_dir,'results.csv')
    # arg_vars = ['model_type','modeling_type', 'test_data_type']
    arg_vars = list(kwargs)
    # if args.model_type =='smp':
    #     arg_vars.extend((list(args.pred_kwargs['smp_args'])))
    arg_vars.extend(list(results))
    arg_vars.extend(list(args.post_process_kwargs))


    # create file
    if not os.path.isfile(res_path):
        with open(res_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(arg_vars)

    # gather variables
    res_dict = {}
    # res_dict.update(args.DataGeneration)
    res_dict.update(results)
    res_dict.update(args.post_process_kwargs)
    # res_dict.update(args.loss_data)
    res_dict.update(kwargs)

    if args.test_on_real_data:
        res_dict.update({'layer_eval_type':args.eval_kwargs['layer_eval_type']})
    # if args.method == 'nn_norm':
    #     res_dict.update(args.net_args)
    # if args.method == 'ista':
    #     res_dict.update(args.dict)
    #     res_dict.update(args.ista)

    # convert dict items to str
    res_dict = [dict([key, str(value)] for key, value in res_dict.items())][0]

    # write results to csv
    df = pd.read_csv(res_path)
    new_data = pd.DataFrame.from_records([res_dict])
    df = df.append(new_data)
    df.to_csv(res_path, index=False)

    # with open(res_dir + '\\exp_args.txt', 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)