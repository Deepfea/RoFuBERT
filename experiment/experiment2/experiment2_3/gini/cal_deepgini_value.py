import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from tqdm import tqdm

from experiment.experiment2.experiment2_3.create_dataset import MyDataset


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


def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)


def cal_deepGini(dataset_name, model_name, dataset_path, load_step_path, save_path):
    test_data_save_path = os.path.join(save_path, model_name, dataset_name)
    if not os.path.exists(test_data_save_path):
        os.makedirs(test_data_save_path)

    test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))

    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)

    print("加载数据集......")
    test_dataset = MyDataset(test_data, tokenizer, dataset_name)
    print("加载数据集完成")

    model_load_path = os.path.join(load_step_path, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')
    model = torch.load(model_load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in tqdm(test_loader):
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = temp_npy
            else:
                total_npy = np.concatenate((total_npy, temp_npy), axis=0)
    output_list = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_arr = np.array(output_list)
    np.save(os.path.join(test_data_save_path, 'test_outputs.npy'), output_arr)
    DeepGini_scores = []
    for num in range(len(output_list)):
        temp_npy = output_list[num]
        temp_score = 1
        for index_num in range(len(temp_npy)):
            temp_score = temp_score - temp_npy[index_num] * temp_npy[index_num]
        DeepGini_scores.append(temp_score)
    test_data['DeepGini_scores'] = DeepGini_scores

    test_data.to_csv(os.path.join(test_data_save_path, 'test.csv'), index=False)
    # print(len(test_data))

    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_data.to_csv(os.path.join(test_data_save_path, 'train.csv'), index=False)
    val_data = pd.read_csv(os.path.join(dataset_path, 'val.csv'))
    val_data.to_csv(os.path.join(test_data_save_path, 'val.csv'), index=False)

if __name__ == '__main__':

    pass