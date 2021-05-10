import csv
import os

import numpy
import pandas as pd
from torch.utils.data import DataLoader
from data_layer.dataset import UltrasoundRealDataset, UltrasoundSimulatedDataset
from data_layer.pre_process_sample import pre_process_sample
from data_layer.pre_process_simulation import pre_process_simulation
from utils.files_operations import load_pickle, is_file_exist


def prepare_data(data=[],
                 learning_model=True,
                 test_on_real_data=True,
                 test_path='./',
                 model_type = 'net',
                 test_gt_path = './',
                 phantom_num=1,
                 with_gaussian_gt=True,
                 ):

    prepared_data = {
        'data': {},
        'dataloaders': {}
    }
    sets = []

    if learning_model:

        sets = ['train', 'val']
        # if not test_on_real_data:
        #     sets.append('test')
        partition = train_test_split(data['target'].shape[0], sets)
        for par_set in sets:
            set_inds = partition[par_set]

            split_set = {'input': data['input'][set_inds, :],
                         'spikes': data['target'][set_inds, :],
                         'spikes_gaus': data['target_gaus'][set_inds, :]if with_gaussian_gt else [],
                         'separated_echoes': data['separated_echoes'][set_inds, :],
                         'echo_inds': data['echo_params']['ind_by_signal'][set_inds][:]
                         }
            prepared_data['data'][par_set] = UltrasoundSimulatedDataset(split_set, model_type, with_gaussian_gt)

    sets.append('test')
    if test_on_real_data:  # test on physical models
        sample_data = UltrasoundRealDataset(test_path, model_type,phantom_num)
        pre_process_sample_data = pre_process_sample(sample_data, phantom_num, test_gt_path)
        prepared_data['data']['test'] = pre_process_sample_data
    else:
        # if 'test' in data:
        data = data['test']
        # else:
        #     data = load_pickle(test_path)

        set = {'input': data['input'],
                     'spikes': data['target'],
                     'spikes_gaus': data['target_gaus'] if with_gaussian_gt else [],
                     'separated_echoes': data['separated_echoes'],
                     'echo_inds': data['echo_params']['ind_by_signal']
                     }
        prepared_data['data']['test'] = UltrasoundSimulatedDataset(set, model_type, with_gaussian_gt)

    for par_set in sets:
        if par_set == 'train':
            prepared_data['dataloaders'][par_set] = DataLoader(prepared_data['data'][par_set], batch_size=300, shuffle=True)
        else:
            prepared_data['dataloaders'][par_set] = DataLoader(prepared_data['data'][par_set], batch_size=300, shuffle=False)

    return prepared_data


def train_test_split(num_samples, sets=['train', 'val', 'test']):
    permuted = numpy.random.permutation(num_samples)

    p1 = int(num_samples * 0.8)
    # p2 = int(num_samples * 0.8)
    partition = {
            'train': permuted[:p1],
            'val': permuted[p1:]
    }
    # if 'test' in sets:
    #     partition['test'] = permuted[p2:]

    return partition


def data2file_old(data_path, images_paths):
    image_dir = os.path.join(data_path, '1d_signals')
    mask_dir = os.path.join(data_path, '1d_echoes')
    with open(os.path.join(data_path, images_paths), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'mask'])
        for p in os.listdir(image_dir):
            image_path = os.path.join(image_dir, p)
            mask_path = os.path.join(mask_dir, p)
            writer.writerow([image_path, mask_path])


def train_test_split_old(data_path, images_paths):
    data = pd.read_csv(os.path.join(data_path, images_paths))
    data = data.iloc[numpy.random.permutation(len(data))]
    p1 = int(len(data) * 0.6)
    p2 = int(len(data) * 0.2)
    train_data, validation_data, test_data = data[:p1], data[p1:p2], data[p2:]

    train_data.to_csv(os.path.join(data_path, 'Train.csv'), index=False)
    validation_data.to_csv(os.path.join(data_path, 'Validation.csv'), index=False)
    test_data.to_csv(os.path.join(data_path, 'Test.csv'), index=False)

# def  save_data_samples_seperately():
#         if save_batches:
#             os.makedirs(args.data_path)
#             os.makedirs(os.path.join(args.data_path, '1d_echoes'))
#             os.makedirs(os.path.join(args.data_path, '1d_signals'))
#             for i in range(len(Data.input)):
#                 with open(os.path.join(args.data_path, '1d_signals', str(i) + '.pkl'), 'wb') as f:
#                     pickle.dump(Data.input[i, :].cpu().numpy(), f, pickle.HIGHEST_PROTOCOL)
#                 with open(os.path.join(args.data_path, '1d_echoes', str(i) + '.pkl'), 'wb') as f:
#                     pickle.dump(Data.target[i, :], f, pickle.HIGHEST_PROTOCOL)
