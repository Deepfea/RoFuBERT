import os.path
import torch
import pandas as pd
from torch import nn
from transformers import BertModel, BertTokenizer
from experiment.experiment2.experiment2_3.evaluate_model import get_adv_data
from experiment.experiment2.experiment2_3.create_dataset import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from experiment.experiment2.experiment2_3.evaluate_model import get_adv_data, evaluate
from experiment.experiment2.experiment2_3.rofubert.get_retrain_data import get_retrain_data
from experiment.experiment2.experiment2_3.rofubert.get_train_data import get_train_data


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

def run_rofubert(labels, dataset_name, model_name, dataset_path, load_fuzz_path, load_experiment2_1_path, save_step_path):
    print(dataset_name)
    print(model_name)
    # get_train_data(dataset_name, model_name, dataset_path, save_step_path)
    # get_retrain_data(labels, dataset_name, model_name, load_fuzz_path, save_step_path)
    # # 手动训练模型
    get_adv_data(model_name, dataset_name, load_experiment2_1_path, save_step_path)
    evaluate(model_name, dataset_name, save_step_path)

if __name__ == '__main__':
    pass
