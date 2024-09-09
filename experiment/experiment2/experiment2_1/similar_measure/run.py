import os

import jieba
import numpy as np
import math

import pandas as pd

from experiment.experiment2.experiment2_1.similar_measure.cal_cos_sim import get_sim_value

def cal_avg_distance(sim_score_arr):
    num = len(sim_score_arr)
    ave_scores = 0
    for i in range(num):
        ave_scores += sim_score_arr[i]
    ave_scores = ave_scores / float(num)
    return ave_scores

def cal_rofubert_data_sim_values(model_name, dataset_name, dataset_path, add_path, save_path):
    print('rofubert:')
    print(dataset_name)
    print(model_name)
    save_path = os.path.join(save_path, model_name, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    load_test_path = os.path.join(dataset_path, 'test.csv')
    test_data = pd.read_csv(load_test_path)
    texts = test_data['text']
    load_add_path = os.path.join(add_path, model_name, dataset_name, 'add.csv')
    add_data = pd.read_csv(load_add_path)
    text_index = add_data['origin']
    mutants = add_data['mutant']
    sim_score_list = []
    for mutant_num in range(len(mutants)):
        temp_str2 = mutants[mutant_num]
        temp_str1 = texts[text_index[mutant_num]]
        sim_score = get_sim_value(temp_str1, temp_str2)
        sim_score_list.append(sim_score)
    sim_score_arr = np.array(sim_score_list)
    print('一共有' + str(len(sim_score_arr)) + '对距离。')
    ave_score = cal_avg_distance(sim_score_arr)
    print('平均余弦相似度为：' + str(ave_score))
    np.save(os.path.join(save_path, 'sim_scores.npy'), sim_score_arr)


def cal_Argot_data_sim_values(model_name, dataset_name, add_path, save_path):
    print('argot:')
    important_num_list = [12, 2, 1]
    if dataset_name == 'weibo':
        important_num = important_num_list[2]
    elif dataset_name == 'news':
        important_num = important_num_list[1]
    else:
        important_num = important_num_list[0]
    print(dataset_name)
    print(model_name)
    save_path = os.path.join(save_path, model_name, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    load_mutant_path = os.path.join(add_path, str(important_num), model_name, dataset_name, 'step1_final_mutants.csv')
    data = pd.read_csv(load_mutant_path)
    texts = data['text']
    mutants = data['mutant']
    sim_score_list = []
    for mutant_num in range(len(mutants)):
        temp_str2 = mutants[mutant_num]
        temp_str1 = texts[mutant_num]
        sim_score = get_sim_value(temp_str1, temp_str2)
        sim_score_list.append(sim_score)
    sim_score_arr = np.array(sim_score_list)
    print('一共有' + str(len(sim_score_arr)) + '对距离。')
    ave_score = cal_avg_distance(sim_score_arr)
    print('平均余弦相似度为：' + str(ave_score))
    np.save(os.path.join(save_path, 'sim_scores.npy'), sim_score_arr)

if __name__ == '__main__':
    pass
