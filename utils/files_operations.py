import errno
import os
import pickle
import json
import stat
import tarfile
import csv
import yaml
import gzip
import shutil
import zipfile

import pandas as pd


def is_file_exist(file_path):
    return os.path.isfile(file_path)


def is_folder_exist(folder_path):
    return os.path.isdir(folder_path)


def get_current_working_directory():
    return os.getcwd()


def get_file_name_from_file_path(file_path):
    return os.path.basename(file_path)


def get_folder_path_from_file_path(file_path):
    return os.path.dirname(file_path)


def make_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def make_folder_for_file_creation_if_not_exists(file_path):
    folder_path = os.path.dirname(file_path)
    if folder_path:
        make_folder(folder_path)


def save_to_pickle(obj, file_path):
    make_folder_for_file_creation_if_not_exists(file_path)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


def write_dict_to_csv_with_pandas(data, file_path, fields_order=None):
    make_folder_for_file_creation_if_not_exists(file_path)
    if type(data[max(data.keys())]) == list:
        expected_length = data[max(data.keys())].__len__()
    else:
        expected_length = 1

    for key in data:
        if data[key].__len__() != expected_length:
            print(
                "Key: " + key + ", Length: " + str(data[key].__len__()) + ", expected length: " + str(expected_length))
            raise ValueError("Keys don't have the same length")

    pd.DataFrame(data).to_csv(file_path, columns=fields_order, index=None)


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        pkl_file = pickle.load(handle)
        handle.close()
    return pkl_file


