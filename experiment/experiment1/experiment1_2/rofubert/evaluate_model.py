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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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



def evaluate(model_name, dataset_name, load_model_path, load_data_data):
    # print(model_name)
    # print(dataset_name)
    token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(token_path)

    adv_data_path = os.path.join(load_data_data, model_name, dataset_name, 'adv.csv')
    adv_data = pd.read_csv(adv_data_path)
    adv_dataset = MyDataset(adv_data, tokenizer, dataset_name)
    print('对抗样本数量：' + str(len(adv_data)))
    model_path = os.path.join(load_model_path, model_name, dataset_name, 'best_' + dataset_name + '_' + model_name + '.pt')
    print(model_path)
    model = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    adv_loader = DataLoader(adv_dataset, batch_size=128)
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in tqdm(adv_loader):
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    result = 1 - total_acc_test / float(len(adv_data))
    print(f'Test Accuracy: {result: .4f}')


if __name__ == '__main__':
    pass


