import os
import pandas as pd
import numpy as np
from torch import nn
from experiment.experiment2.experiment2_1.RoFuBERT.cal_test_output import cal_output
from experiment.experiment2.experiment2_1.RoFuBERT.cut_seg import cut_seg
from experiment.experiment2.experiment2_1.RoFuBERT.cal_word_scores import cal_word_scores, select_segs
from experiment.experiment2.experiment2_1.RoFuBERT.generate_mutants import generate_mutant1
from experiment.experiment2.experiment2_1.RoFuBERT.select_mutants import select_mutants


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

def load_test_len(dataset_name, model_name, load_step1_path):
    load_test_path = os.path.join(load_step1_path, model_name, dataset_name)
    test_data = pd.read_csv(os.path.join(load_test_path, 'step0_test.csv'))
    return len(test_data)

def generate_all_samples(dataset_name, model_name, load_step_path, important_num):
    load_step1_path = os.path.join(load_step_path, model_name + '_' + dataset_name, str(important_num))
    if not os.path.exists(load_step1_path):
        os.makedirs(load_step1_path)
    select_segs(dataset_name, model_name, load_step_path, load_step1_path, important_num)
    generate_mutant1(dataset_name, model_name, load_step_path, load_step1_path)

def get_n_all_samples(dataset_name, model_name, load_step1_path, max_num, interval):
    print(dataset_name)
    print(model_name)
    important_num = 0
    while important_num < max_num:
        important_num += interval
        print(important_num)
        generate_all_samples(dataset_name, model_name, load_step1_path, important_num)

def get_all_words_and_score(labels, dataset_name, model_name, dataset_path, load_step0_path, load_step1_path):
    print(dataset_name)
    print(model_name)
    cal_output(dataset_name, model_name, dataset_path, load_step0_path, load_step1_path)
    cut_seg(labels, dataset_name, model_name, load_step1_path)
    cal_word_scores(labels, dataset_name, model_name, load_step0_path, load_step1_path)

def get_final_mutants(dataset_name, model_name, load_model_path, load_step_path, max_num, interval):
    print(dataset_name)
    print(model_name)
    important_num = 0
    while important_num < max_num:
        important_num += interval
        print(important_num)
        load_step1_path = os.path.join(load_step_path, model_name + '_' + dataset_name, str(important_num))
        select_mutants(dataset_name, model_name, load_model_path, load_step_path, load_step1_path)

def count_num(dataset_name, model_name, load_step_path, max_num, interval):
    print(dataset_name)
    print(model_name)
    important_num = 0
    while important_num < max_num:
        important_num += interval
        print(important_num)
        load_step1_path = os.path.join(load_step_path, model_name + '_' + dataset_name, str(important_num))
        final_mutants = pd.read_csv(os.path.join(load_step1_path, model_name, dataset_name, 'step1_final_mutants.csv'))
        index_list = np.array(final_mutants['belong'])
        index_list = np.unique(index_list)
        print(len(index_list))

def count_avg_seg_num(dataset_name, model_name, load_step_path):
    print(dataset_name)
    print(model_name)
    load_step1_path = os.path.join(load_step_path, model_name, dataset_name, 'step0_test_final_segs.npy')
    seg_arr = np.load(load_step1_path, allow_pickle=True)
    # print(len(seg_arr))
    seg_num = 0
    for num in range(len(seg_arr)):
        # print(len(seg_arr[num]))
        seg_num += len(seg_arr[num])
    # print(seg_num)
    avg_num = seg_num / float(len(seg_arr))
    print(avg_num)


if __name__ == '__main__':

    pass







