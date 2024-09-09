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
from experiment.experiment2.experiment2_2.create_dataset import MyDataset

# model_name = 'bert_base_chinese'
# dataset_name = 'cail'
# load_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
# tokenizer = BertTokenizer.from_pretrained(load_path)
# model = BertModel.from_pretrained(load_path)
# labels = ['盗窃', '危险驾驶', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品',
#               '容留他人吸毒', '寻衅滋事', '抢劫', '非法持有毒品', '滥伐林木']

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

def evaluate_ori(model_name, dataset_name, load_path):
    print('Origin:')
    # print(model_name)
    # print(dataset_name)
    token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(token_path)

    adv_data_path = os.path.join(load_path, model_name, dataset_name, 'adv.csv')
    adv_data = pd.read_csv(adv_data_path)
    adv_dataset = MyDataset(adv_data, tokenizer, dataset_name)
    print('对抗样本数量：' + str(len(adv_data)))

    model_load_path = '/media/usr/external/home/usr/project/project2_data/step0_train_and_test'
    model_path = os.path.join(model_load_path, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')
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
    print(f'Attack Success Rate: {result: .4f}')

def evaluate(model_name, dataset_name, load_path):
    # print(model_name)
    # print(dataset_name)
    token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(token_path)

    adv_data_path = os.path.join(load_path, model_name, dataset_name, 'adv.csv')
    adv_data = pd.read_csv(adv_data_path)
    adv_dataset = MyDataset(adv_data, tokenizer, dataset_name)
    print('对抗样本数量：' + str(len(adv_data)))

    coverage_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = []
    for rate_num in range(len(coverage_rate_list)):
        temp_coverage_rate = coverage_rate_list[rate_num]
        print(temp_coverage_rate)
        model_path = os.path.join(load_path, model_name, dataset_name, 'retrain_model', str(temp_coverage_rate), 'best_' + dataset_name + '_' + model_name + '.pt')
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
        results.append(result)
    results = np.array(results)
    np.save(os.path.join(load_path, model_name, dataset_name, 'evaluate_results.npy'), results)

def set_important_num(dataset_name):
    if dataset_name == 'cail':
        important_num = 12
    elif dataset_name == 'news':
        important_num = 2
    else:
        important_num = 1
    return important_num

def get_adv_data(model_name, dataset_name, load_path, save_path):
    important_num = set_important_num(dataset_name)
    load_data_path = os.path.join(load_path, model_name + '_' + dataset_name, str(important_num), model_name, dataset_name, 'step1_final_mutants.csv')
    mutant = pd.read_csv(load_data_path)
    text_list = mutant['mutant']
    label_list = mutant['label']
    merge_dt_dict = {'text': text_list, 'label': label_list}
    adv_data = pd.DataFrame(merge_dt_dict)
    save_data_path = os.path.join(save_path, model_name, dataset_name, 'adv.csv')
    adv_data.to_csv(save_data_path, index=False)


if __name__ == '__main__':
    pass


