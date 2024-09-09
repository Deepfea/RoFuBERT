import os.path
import torch
import pandas as pd
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment.experiment1.experiment1_2.rofubert.count_modify_rate import count_modify_rate
from experiment.experiment1.experiment1_2.rofubert.evaluate_model import evaluate
from experiment.experiment1.experiment1_2.rofubert.get_adv import get_adv_data


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

def run_rofubert(dataset_name, model_name, dataset_path, load_model_path, load_experiment2_1_path, load_step_fuzz_path, save_step_path):
    print(dataset_name)
    print(model_name)
    # # 手动训练模型
    # get_adv_data(model_name, dataset_name, load_experiment2_1_path, save_step_path)
    # evaluate(model_name, dataset_name, load_model_path, save_step_path)
    count_modify_rate(dataset_name, model_name, load_step_fuzz_path)

if __name__ == '__main__':
    pass
