from torch.optim import Adam
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch import nn
from experiment.experiment2.experiment2_3.create_dataset import MyDataset

def set_important_num(dataset_name):
    if dataset_name == 'cail':
        important_num = 12
    elif dataset_name == 'news':
        important_num = 2
    else:
        important_num = 1
    return important_num

def get_adv_data(model_name, dataset_name, load_path, save_path):
    temp_path = os.path.join(save_path, model_name, dataset_name)
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    important_num = set_important_num(dataset_name)
    load_data_path = os.path.join(load_path, model_name + '_' + dataset_name, str(important_num), model_name, dataset_name, 'step1_final_mutants.csv')
    mutant = pd.read_csv(load_data_path)
    text_list = mutant['mutant']
    label_list = mutant['label']
    merge_dt_dict = {'text': text_list, 'label': label_list}
    adv_data1 = pd.DataFrame(merge_dt_dict)

    important_num_list = [1, 2, 12]
    if dataset_name == 'weibo':
        important_num = important_num_list[0]
    elif dataset_name == 'news':
        important_num = important_num_list[1]
    else:
        important_num = important_num_list[2]
    load_add_path = os.path.join(load_path, 'Argot', str(important_num), model_name, dataset_name)
    mutant = pd.read_csv(os.path.join(load_add_path, 'step1_final_mutants.csv'))
    text_list = mutant['mutant']
    label_list = mutant['label']
    merge_dt_dict = {'text': text_list, 'label': label_list}
    adv_data2 = pd.DataFrame(merge_dt_dict)

    adv_data = pd.concat([adv_data1, adv_data2], ignore_index=True)
    save_data_path = os.path.join(save_path, model_name, dataset_name, 'adv.csv')
    adv_data.to_csv(save_data_path, index=False)
    print(len(adv_data1))
    print(len(adv_data2))
    print(len(adv_data))


if __name__ == '__main__':
    model_name = 'bert_base_chinese'
    dataset_name = 'weibo'
    load_path = '/media/usr/external/home/usr/project/project2_data/epxeriment/experiment2/experiment2_1'
    save_path = '/media/usr/external/home/usr/project/project2_data/epxeriment/experiment1/experiment1_2/rofubert'
    get_adv_data(model_name, dataset_name, load_path, save_path)



