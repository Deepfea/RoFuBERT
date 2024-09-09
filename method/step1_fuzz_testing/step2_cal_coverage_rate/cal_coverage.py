import numpy as np
import os
import pandas as pd

def cal_coverage(load_step_path, model_name, dataset_name):
    if dataset_name == 'cail':
        bucket_num = 200
    elif dataset_name == 'news':
        bucket_num = 200
    else:
        bucket_num = 200
    load_value_path = os.path.join(load_step_path, model_name, dataset_name)
    train_values = np.load(os.path.join(load_value_path, 'step2_train_lime_scores.npy'))
    bucket_arr = np.zeros(bucket_num + 1, dtype='int')
    interval = 1.0 / float(bucket_num)
    for train_values_num in range(len(train_values)):
        temp_value = train_values[train_values_num]
        index = temp_value / interval
        bucket_arr[int(index)] = 1
        # print(temp_value)
    np.save(os.path.join(load_value_path, 'step2_current_coverage.npy'), bucket_arr)

def cal_gini(temp_npy):
    temp_score = 1
    for index_num in range(len(temp_npy)):
        temp_score = temp_score - temp_npy[index_num] * temp_npy[index_num]
    return temp_score

def cal_add_coverage(load_step_path, model_name, dataset_name, counter):
    # 更新add集合
    load_value_path = os.path.join(load_step_path, model_name, dataset_name)
    data = pd.read_csv(os.path.join(load_value_path, 'step1_seed_mutants.csv'))
    add_save_file = os.path.join(load_value_path, 'add.csv')
    if os.path.exists(add_save_file):
        temp_df = pd.read_csv(add_save_file)
        data = pd.concat([data, temp_df], ignore_index=True)
    data.to_csv(add_save_file, index=False)

    # 更新覆盖率并生成新的覆盖变异体集合
    if dataset_name == 'cail':
        bucket_num = 200
    elif dataset_name == 'news':
        bucket_num = 200
    else:
        bucket_num = 200
    interval = 1.0 / float(bucket_num)
    mutant_values = np.load(os.path.join(load_value_path, 'step2_mutants_lime_scores.npy'))
    bucket_arr = np.load(os.path.join(load_value_path, 'step2_current_coverage.npy'))
    mutant_output = np.load(os.path.join(load_value_path, 'step1_seed_mutants_outputs.npy'))
    text_list = []
    label_list = []
    deepgini_list = []
    counter_list = []
    output_list = []
    ori_num_list = []
    for mutant_values_num in range(len(mutant_values)):
        temp_value = mutant_values[mutant_values_num]
        index = temp_value / interval
        if bucket_arr[int(index)] == 0:
            bucket_arr[int(index)] = 1
            gini_value = cal_gini(mutant_output[mutant_values_num])
            output_list.append(mutant_output[mutant_values_num])
            text_list.append(data.loc[mutant_values_num, 'mutant'])
            label_list.append(data.loc[mutant_values_num, 'label'])
            deepgini_list.append(gini_value)
            counter_list.append(counter)
            ori_num_list.append(data.loc[mutant_values_num, 'origin'])
    merge_dt_dict = {'text': text_list, 'label': label_list, 'DeepGini_scores': deepgini_list,
                     'counter': counter_list, 'origin': ori_num_list}
    data_df = pd.DataFrame(merge_dt_dict)
    data_df.to_csv(os.path.join(load_value_path, 'step2_next_turn_seeds.csv'), index=False)

    output_arr = np.array(output_list)

    np.save(os.path.join(load_value_path, 'step2_next_turn_seeds_outputs.npy'), output_arr)

    np.save(os.path.join(load_value_path, 'step2_current_coverage.npy'), bucket_arr)

if __name__ == '__main__':

    pass







