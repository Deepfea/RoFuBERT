import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn

from method.step1_fuzz_testing.create_dataset import MyDataset


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

def cal_word_scores(labels, dataset_name, model_name, load_model_path ,load_step_path, important_num):
    load_seed_path = os.path.join(load_step_path, model_name, dataset_name)
    seeds = pd.read_csv(os.path.join(load_seed_path, 'step0_test_seeds.csv'))
    seg_list = np.load(os.path.join(load_seed_path, 'step0_test_seeds_segs.npy'), allow_pickle=True)
    ori_output_list = np.load(os.path.join(load_seed_path, 'step0_test_seeds_outputs.npy'), allow_pickle=True)
    belong_list = []
    str_list = []
    fact_list = []
    label_list = []
    remove_flag = ['。', '.', '，', ',', '；', ';', '？', '?',
                   '、', '！', '!', '“', '”', '"', '《', '》',
                   '~', '<', '>', '：', ':', '%', '/',
                   '（', '(', '）', ')', '-', '—', '[', ']', '【', '】', '@', '·']
    for data_num in range(len(seeds)):
        tem_segs = list(set(seg_list[data_num]))
        ori_fact = seeds.loc[data_num, 'text']
        label = seeds.loc[data_num, 'label']
        for seg_num in range(len(tem_segs)):
            temp_seg = tem_segs[seg_num]
            if temp_seg in remove_flag:
                continue
            # print(temp_seg)
            temp_fact = ori_fact.replace(temp_seg, '')
            belong_list.append(data_num)
            str_list.append(temp_seg)
            fact_list.append(temp_fact)
            label_list.append(label)
    merge_dt_dict = {'belong': belong_list, 'str': str_list, 'text': fact_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    # print(data_df)
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    test_dataset = MyDataset(data_df, tokenizer, dataset_name)

    model_load_path = os.path.join(load_model_path, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')
    model = torch.load(model_load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
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
    # print(len(output_list))

    importance_value = []
    for data_num in range(len(output_list)):
        temp_output = np.array(output_list[data_num])
        temp_label = data_df.loc[data_num, 'label']
        temp_value = ori_output_list[data_df.loc[data_num, 'belong']][temp_label] - temp_output[temp_label]
        importance_value.append(temp_value)
    data_df['importance_value'] = importance_value
    df_filtered = data_df[data_df['importance_value'] > -100].reset_index(drop=True)
    # print(len(df_filtered))
    # print(seg_list)
    seg_list = []
    seg_score = []
    for class_num in range(len(labels)):
        seg_list.append([])
        seg_score.append([])
    for df_filtered_num in range(len(df_filtered)):
        class_name = df_filtered.loc[df_filtered_num, 'belong']
        seg_list[class_name].append(df_filtered.loc[df_filtered_num, 'str'])
        seg_score[class_name].append(df_filtered.loc[df_filtered_num, 'importance_value'])
    # print(seg_list)
    # print(seg_score)
    seg_list, seg_score = select_seg(seg_list, seg_score, important_num)
    seg_arr = np.array(seg_list)
    np.save(os.path.join(load_seed_path, 'step1_test_seeds_segs.npy'), seg_arr)



if __name__ == '__main__':

    pass