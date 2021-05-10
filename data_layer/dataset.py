from copy import deepcopy

import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
from utils.HdfDataHandler import HdfData
import pickle


class UltrasoundSimulatedDataset(Dataset):

    def __init__(self, data, model_type ='net',with_gaussian_gt=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """

        self.model_type = model_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.model_type == 'net':
            if with_gaussian_gt:
                raw_gt = data['spikes_gaus']
            else:
                raw_gt = data['spikes']

            self.input = torch.tensor(data['input']).reshape(data['input'].shape[0], 1, data['input'].shape[1], 1).to(self.device)
            self.gt = torch.tensor(raw_gt).to(self.device)
        else:
            self.gt = []
            self.input = data['input']
        # self.input = data['input']

        # self.signals = data['input']
        self.spikes = data['spikes']
        self.separated_echoes=data['separated_echoes']
        self.echo_inds = data['echo_inds']

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):

        if self.model_type =='net':
            X = self.input[idx,:,:,:]
        else:
            X = self.input[idx, :]

        y = self.gt[idx,:]

        return {'input':X, 'gt':y}

    def get_signal(self,idx):
        signal = self.input[idx,:]
        spike = self.spikes[idx,:]
        separated_echoes = self.separated_echoes[idx,:,:]

        return signal, spike, separated_echoes


class UltrasoundRealDataset(Dataset, HdfData):

    def __init__(self, root_dir,model_type='',phantom_num=2):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string: Directory with all the images.
        """
        #super(Hdf_data,self)
        # self.image_frame = pd.read_csv(csv_file, skiprows=0)
        self.root_dir = root_dir
        super().__init__(root_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.phantom_num = phantom_num
        if self.model_type == 'net':
            self.input = torch.tensor(self.a_scan_mat).reshape(self.a_scan_mat.shape[0], 1, self.a_scan_mat.shape[1], 1).to(self.device)
        # self.input = input

    def __len__(self):
        return self.a_scan_mat.shape[0]

    def set_input(self, pre_processed_input):
        self.input = pre_processed_input
        if self.model_type == 'net':
            self.input = torch.tensor(deepcopy(self.a_scan_mat)).reshape(self.a_scan_mat.shape[0], 1, self.a_scan_mat.shape[1], 1).to(self.device).float()
        else:
            self.input = deepcopy(self.a_scan_mat)

    def set_cscan_gt(self,cscan_gt):

        self.c_scan_gt = cscan_gt

    def __getitem__(self, idx):
        if self.model_type == 'net':
            return {'input':self.input[idx,:,:,:]}
        else:
            return {'input':self.input[idx,:]}


class SimulatedDataOld(Dataset):
    def __init__(self, list_IDs, root_dir):

        self.root_dir = root_dir
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        mask_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 1])

        image = pickle.load(open(img_name, 'rb'))
        if len(image.shape) > 1:
            X = np.resize(image,[1, image.shape[0],image.shape[1]])
        else:
            X = np.resize(image, [1, image.shape[0],1])
        # Load data and get label
        mask = pickle.load(open(mask_name, 'rb'))
        if len(mask.shape) > 1:
            y = np.resize(image,[1, image.shape[0],image.shape[1]])
        else:
            y = np.resize(image, [1, image.shape[0],1])

        # example
        # ID = self.list_IDs[index]
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]

        return X, y


class LungCTDataset(Dataset):
    """LungCT dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.image_frame = pd.read_csv(csv_file, skiprows=0)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.ix[idx, 0])
        mask_name = os.path.join(self.root_dir, self.image_frame.ix[idx, 1])

        image = cv2.imread(img_name, 0)
        image.resize(32, 32)
        image = image.reshape((1, 32, 32))
        mask = cv2.imread(mask_name, 0)
        mask.resize(32, 32)
        mask = mask.reshape((1, 32, 32))
        sample = {'image': image, 'mask': mask}
        return sample

