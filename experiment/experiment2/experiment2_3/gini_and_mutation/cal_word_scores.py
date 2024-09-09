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

def select_segs(dataset_name, model_name, load_step_path, important_num):
    load_seg_path = os.path.join(load_step_path, model_name, dataset_name)
    seg_arr = np.load(os.path.join(load_seg_path, 'final_segs.npy'), allow_pickle=True)
    value_arr = np.load(os.path.join(load_seg_path, 'final_seg_importance_values.npy'), allow_pickle=True)
    imp_seg_list, imp_sco_list = select_seg(seg_arr, value_arr, important_num)
    save_seg_path = os.path.join(load_seg_path, str(important_num))
    if not os.path.exists(save_seg_path):
        os.makedirs(save_seg_path)
    np.save(os.path.join(save_seg_path, 'select_segs.npy'), np.array(imp_seg_list))
    np.save(os.path.join(save_seg_path, 'select_segs_values.npy'), np.array(imp_sco_list))

def select_seg(seg_list, seg_score, important_num):
    imp_list = []
    imp_score = []
    for num in range(len(seg_list)):
        temp_list = seg_list[num]
        temp_score = seg_score[num]
        if len(temp_list) < important_num:
            imp_list.append(temp_list)
            imp_score.append(temp_score)
        else:
            temp_score = np.array(temp_score)
            temp_index = np.argsort(-temp_score)
            imp_temp_list = []
            imp_temp_score = []
            # print(temp_score)
            for imp_num in range(important_num):
                imp_index = temp_index[imp_num]
                imp_temp_list.append(temp_list[imp_index])
                imp_temp_score.append(temp_score[imp_index])
            # print(imp_temp_score)
            imp_list.append(imp_temp_list)
            imp_score.append(imp_temp_score)

    # print(imp_list)
    return imp_list, imp_score

def cal_word_scores(labels, dataset_name, model_name, load_model_path ,load_step_path):
    load_seed_path = os.path.join(load_step_path, model_name, dataset_name)
    test_data = pd.read_csv(os.path.join(load_seed_path, 'final_seed.csv'))
    seg_list = np.load(os.path.join(load_seed_path, 'seed_segs.npy'), allow_pickle=True)
    ori_output_list = np.load(os.path.join(load_seed_path, 'final_seed_outputs.npy'), allow_pickle=True)
    print(len(test_data))
    str_list = []
    fact_list = []
    label_list = []
    belong_list = []
    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
    final_segs = []
    for data_num in range(len(test_data)):
        temp_final_segs = []
        tem_segs = list(set(seg_list[data_num]))
        ori_fact = test_data.loc[data_num, 'text']
        label = test_data.loc[data_num, 'label']
        for seg_num in range(len(tem_segs)):
            temp_seg = tem_segs[seg_num]
            if temp_seg in remove_flag:
                continue
            # print(temp_seg)
            temp_final_segs.append(temp_seg)
            temp_fact = ori_fact.replace(temp_seg, '')
            belong_list.append(data_num)
            str_list.append(temp_seg)
            fact_list.append(temp_fact)
            label_list.append(label)
        final_segs.append(temp_final_segs)
    merge_dt_dict = {'belong': belong_list, 'str': str_list, 'text': fact_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    np.save(os.path.join(load_seed_path, 'final_segs.npy'), np.array(final_segs))
    print(len(data_df))

    print("加载数据集......")
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    test_dataset = MyDataset(data_df, tokenizer, dataset_name)
    print("加载数据集完成")

    model_load_path = os.path.join(load_model_path, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')
    model = torch.load(model_load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=256)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in tqdm(test_loader):
            temp_len = len(test_label)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            if temp_len == 1:
                temp_list = []
                temp_list.append(temp_npy)
                temp_npy = np.array(temp_list)
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

    importance_value = []
    for data_num in range(len(output_list)):
        temp_output = np.array(output_list[data_num])
        temp_label = data_df.loc[data_num, 'label']
        temp_value = ori_output_list[data_df.loc[data_num, 'belong']][temp_label] - temp_output[temp_label]
        importance_value.append(temp_value)
    data_df['importance_value'] = importance_value

    value_list = []
    data_num_left = 0
    data_num_right = 0
    for seg_num in range(len(final_segs)):
        data_num_right += len(final_segs[seg_num])
        temp_value_list = []
        while data_num_left < data_num_right:
            temp_value_list.append(data_df.loc[data_num_left, 'importance_value'])
            # print(data_num_left)
            data_num_left += 1
        value_list.append(temp_value_list)
    value_arr = np.array(value_list)
    np.save(os.path.join(load_seed_path, 'final_seg_importance_values.npy'), value_arr)
    print(len(value_arr))

if __name__ == '__main__':

    pass
