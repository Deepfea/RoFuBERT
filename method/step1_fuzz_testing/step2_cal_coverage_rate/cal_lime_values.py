import numpy as np
import os
from transformers import BertTokenizer
import pandas as pd
from torch import nn

from method.step1_fuzz_testing.lime_for_text import cal_lime_value

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

def convert_value(result_list):
    score_list = []
    for result_num in range(len(result_list)):
        temp_result = result_list[result_num]
        all_num = len(temp_result)
        positive_word_num = 0
        for word_num in range(all_num):
            if temp_result[word_num][1] > 0:
                positive_word_num += 1
        temp_score = positive_word_num / float(all_num)
        score_list.append(temp_score)
    score_arr = np.array(score_list)
    return score_arr
def cal_ori_lime_value(dataset_name, model_name, labels, load_dataset_path, load_model_path, load_step_path):
    save_path = os.path.join(load_step_path, model_name, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("加载数据集......")
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    train_data = pd.read_csv(os.path.join(load_dataset_path, 'train.csv'))
    print("加载数据集完成")

    results = cal_lime_value(tokenizer, load_model_path, model_name, dataset_name, labels, train_data)
    np.save(os.path.join(save_path, 'step2_train_lime_results.npy'), np.array(results))
    scores = convert_value(results)

    np.save(os.path.join(save_path, 'step2_train_lime_scores.npy'), scores)


def cal_mutant_lime_value(dataset_name, model_name, labels, load_model_path, load_step_path):
    load_seed_path = os.path.join(load_step_path, model_name, dataset_name)

    print("加载数据集......")
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    data = pd.read_csv(os.path.join(load_seed_path, 'step1_seed_mutants.csv'))
    print("加载数据集完成")

    results = cal_lime_value(tokenizer, load_model_path, model_name, dataset_name, labels, data)
    # print(results)
    np.save(os.path.join(load_seed_path, 'step2_mutant_results.npy'), np.array(results))
    scores = convert_value(results)
    # print(scores)
    np.save(os.path.join(load_seed_path, 'step2_mutants_lime_scores.npy'), scores)

if __name__ == '__main__':
    pass







