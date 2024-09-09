import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from tqdm import tqdm
from experiment.experiment2.experiment2_1.RoFuBERT.create_dataset import mutantDataset


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

def select_mutants(dataset_name, model_name, load_model_path, load_step_path, important_num):
    load_test_path = os.path.join(load_step_path, model_name, dataset_name)
    load_mutant_path = os.path.join(load_test_path, str(important_num))
    mutants = pd.read_csv(os.path.join(load_mutant_path, 'seed_mutants.csv'))
    mutants_output = cal_output(dataset_name, model_name, load_model_path, mutants)
    temp_list = []
    final_mutants_output = []
    for output_num in range(len(mutants_output)):
        temp_mutant_output = mutants_output[output_num]
        temp_mutant_label = np.argmax(temp_mutant_output)
        ori_label = mutants.loc[output_num, 'label']
        if temp_mutant_label == ori_label:
            temp_list.append(output_num)
        else:
            final_mutants_output.append(temp_mutant_output)
    final_mutants = mutants.drop(temp_list).reset_index(drop=True)
    final_mutants_output = np.array(final_mutants_output)
    print(len(final_mutants))
    print(len(final_mutants_output))
    final_mutants.to_csv(os.path.join(load_mutant_path, 'final_mutants.csv'), index=False)
    np.save(os.path.join(load_mutant_path, 'final_mutants_output.npy'), final_mutants_output)

def cal_output(dataset_name, model_name, load_path, data):
    model_load_path = os.path.join(load_path, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')
    model = torch.load(model_load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    dataset = mutantDataset(data, tokenizer, dataset_name)
    print(len(dataset))
    test_loader = DataLoader(dataset, batch_size=256)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in tqdm(test_loader):
            temp_len = len(test_label)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            # print(temp_npy.shape)
            if temp_len == 1:
                temp_list = []
                temp_list.append(temp_npy)
                temp_npy = np.array(temp_list)
            # print(temp_npy.shape)
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
    return output_arr

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

if __name__ == '__main__':

    pass
