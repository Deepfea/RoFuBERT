import os

import numpy as np
import pandas as pd
from transformers import BertTokenizer
from method.step1_fuzz_testing.lime_for_text import cal_lime_value

def get_and_save_retrain_data(dataset_name, model_name, fuzz_path, save_path):
    if dataset_name == 'cail':
        dataset_path = '/media/usr/external/home/usr/project/project2_data/dataset/CAIL2018_ALL_DATA'
    elif dataset_name == 'news':
        dataset_path = '/media/usr/external/home/usr/project/project2_data/dataset/THUCNews'
    else:
        dataset_path = '/media/usr/external/home/usr/project/project2_data/dataset/weibo'
    print(dataset_name)
    print(model_name)

    train_data_path = os.path.join(dataset_path, 'train.csv')
    train_data = pd.read_csv(train_data_path)

    val_data_path = os.path.join(dataset_path, 'val.csv')
    val_data = pd.read_csv(val_data_path)

    add_data_path = os.path.join(fuzz_path, model_name, dataset_name, 'add.csv')
    add_data = pd.read_csv(add_data_path)

    save_data_path = os.path.join(save_path, model_name, dataset_name)
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    retrain_data_text = []
    retrain_data_text += list(train_data['text'])
    retrain_data_text += list(add_data['mutant'])
    retrain_data_label = []
    retrain_data_label += list(train_data['label'])
    retrain_data_label += list(add_data['label'])

    merge_dt_dict = {'text': retrain_data_text, 'label': retrain_data_label}
    retrain_data = pd.DataFrame(merge_dt_dict)
    print(len(retrain_data))
    retrain_data.to_csv(os.path.join(save_data_path, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(save_data_path, 'val.csv'), index=False)

def convert_value(result_list):
    score_list = []
    for result_num in range(len(result_list)):
        temp_result = result_list[result_num]
        print(temp_result)
        all_num = len(temp_result)
        positive_word_num = 0
        for word_num in range(all_num):
            if temp_result[word_num][1] > 0:
                positive_word_num += 1
        temp_score = positive_word_num / float(all_num)
        score_list.append(temp_score)
    score_arr = np.array(score_list)
    print(score_arr)
    return score_arr

def collect_all_retrain_score(dataset_name, model_name, labels, train_model_path, fuzz_path, save_path):
    train_score_path = os.path.join(fuzz_path, model_name, dataset_name, 'step2_train_lime_scores.npy')
    train_lime_scores = np.load(train_score_path)
    add_data_path = os.path.join(fuzz_path, model_name, dataset_name, 'add.csv')
    add_data = pd.read_csv(add_data_path)
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    add_results = cal_lime_value(tokenizer, train_model_path, model_name, dataset_name, labels, add_data)
    scores = convert_value(add_results)
    all_scores = np.concatenate((train_lime_scores, scores), axis=0)
    print(len(all_scores))
    np.save(os.path.join(save_path, model_name, dataset_name, 'lime_scores.npy'), all_scores)

if __name__ == '__main__':

    pass
