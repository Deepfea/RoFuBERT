import numpy as np
import os
import pandas as pd
import jieba

def select_seed_and_segmentation(labels, dataset_name, model_name,load_step_path):
    load_value_path = os.path.join(load_step_path, model_name, dataset_name)
    test_data = pd.read_csv(os.path.join(load_value_path, 'step0_test.csv'))
    test_output = np.load(os.path.join(load_value_path, 'step0_test_outputs.npy'))
    text_list = []
    label_list = []
    value_list = []
    output_list = []
    num_list = []
    counter_list = []
    ori_num_list = []
    for class_num in range(len(labels)):
        temp_score = 0
        temp_text = ''
        temp_output = test_output[0]
        temp_num = -1
        temp_counter = -1
        temp_ori_num = -1
        for data_num in range(len(test_data)):
            if test_data.loc[data_num, 'label'] != class_num:
                continue
            if test_data.loc[data_num, 'DeepGini_scores'] > temp_score:
                temp_score = test_data.loc[data_num, 'DeepGini_scores']
                temp_text = test_data.loc[data_num, 'text']
                temp_output = test_output[data_num]
                temp_num = data_num
                temp_counter = test_data.loc[data_num, 'counter']
                temp_ori_num = test_data.loc[data_num, 'origin']
        if temp_num != -1:
            num_list.append(temp_num)
            text_list.append(temp_text)
            label_list.append(class_num)
            value_list.append(temp_score)
            output_list.append(temp_output)
            counter_list.append(temp_counter)
            ori_num_list.append(temp_ori_num)
    merge_dt_dict = {'data_num': num_list, 'text': text_list, 'label': label_list, 'gini_score': value_list, 'counter': counter_list, 'origin': ori_num_list}
    data_df = pd.DataFrame(merge_dt_dict)
    # print(data_df)
    # print(value_list)
    seg_list = []
    for data_num in range(len(data_df)):
        temp_string = data_df.loc[data_num, 'text']
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        seg_list.append(cut_result)
    seg_arr = np.array(seg_list)
    output_arr = np.array(output_list)
    # print(output_arr)
    np.save(os.path.join(load_value_path, 'step0_test_seeds_segs.npy'), seg_arr)
    np.save(os.path.join(load_value_path, 'step0_test_seeds_outputs.npy'), output_arr)
    data_df.to_csv(os.path.join(load_value_path, 'step0_test_seeds.csv'), index=False)

if __name__ == '__main__':

    pass