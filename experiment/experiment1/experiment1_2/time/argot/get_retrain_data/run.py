import os.path
import torch
import pandas as pd
from torch import nn
import time
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment.experiment1.experiment1_2.time.argot.get_retrain_data.get_retrain_data import get_retrain_data
from experiment.experiment1.experiment1_2.time.argot.get_retrain_data.get_train_data import get_train_data


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, len(labels))
        self.relu = nn.ReLU()
    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

def run_argot(labels, dataset_name, model_name, dataset_path, load_experiment1_2_path,save_step_path):
    print(dataset_name)
    print(model_name)
    get_train_data(dataset_name, model_name, dataset_path, save_step_path)
    get_retrain_data(dataset_name, model_name, load_experiment1_2_path, save_step_path)
    # # 手动训练模型

if __name__ == '__main__':
    pass