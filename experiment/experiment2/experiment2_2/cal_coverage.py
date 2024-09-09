import numpy as np
import os
import pandas as pd
from method.step1_fuzz_testing.step2_cal_coverage_rate.set_bucket_num import set_coverage_bucket

def cal_coverage(save_path, model_name, dataset_name):
    bucket_num = set_coverage_bucket(dataset_name)
    load_value_path = os.path.join(save_path, model_name, dataset_name, 'lime_scores.npy')
    values = np.load(load_value_path)
    bucket_list = []
    interval = 1.0 / float(bucket_num)
    for values_num in range(len(values)):
        temp_value = values[values_num]
        index = int(temp_value / interval)
        bucket_list.append(index)
    retrain_data_path = os.path.join(save_path, model_name, dataset_name, 'train.csv')
    retrain_data = pd.read_csv(retrain_data_path)
    retrain_data['coverage_index'] = bucket_list
    retrain_data.to_csv(os.path.join(save_path, model_name, dataset_name, 'all_retrain.csv'), index=False)
    print(len(retrain_data))

def save_coverage_data(dataset_name, model_name, save_path):

    bucket_num = set_coverage_bucket(dataset_name)

    all_data_path = os.path.join(save_path, model_name, dataset_name, 'all_retrain.csv')
    all_data = pd.read_csv(all_data_path)
    index_list = list(all_data['coverage_index'])
    num = 0
    total_num = 0
    while num < bucket_num + 1:
        current_text_list = []
        current_label_list = []
        current_index_list = []
        for temp_num in range(len(index_list)):
            if index_list[temp_num] == num:
                current_text_list.append(all_data.loc[temp_num, 'text'])
                current_label_list.append(all_data.loc[temp_num, 'label'])
                current_index_list.append(all_data.loc[temp_num, 'coverage_index'])
        total_num += len(current_index_list)
        # print(total_num)
        if len(current_index_list) == 0:
            num += 1
            continue
        merge_dt_dict = {'text': current_text_list, 'label': current_label_list, 'coverage_index': current_index_list}
        current_data = pd.DataFrame(merge_dt_dict)
        coverage_path = os.path.join(save_path, model_name, dataset_name, 'coverage')
        if not os.path.exists(coverage_path):
            os.makedirs(coverage_path)
        current_data.to_csv(os.path.join(coverage_path, str(num) + '_retrain.csv'), index=False)
        num += 1
def rank_coverage_data(dataset_name, model_name, save_path):
    load_csv_name_path = os.path.join(save_path, model_name, dataset_name, 'coverage')
    file_list = os.listdir(load_csv_name_path)
    data_num_list = []
    for file_name in file_list:
        file_path = os.path.join(load_csv_name_path, file_name)
        temp_data = pd.read_csv(file_path)
        data_num_list.append(len(temp_data))
    data_num_arr = np.array(data_num_list)
    sort_index = np.argsort(data_num_arr)[::-1]
    save_csv_name_path = os.path.join(save_path, model_name, dataset_name, 'sort_coverage')
    if not os.path.exists(save_csv_name_path):
        os.makedirs(save_csv_name_path)
    for num in range(len(sort_index)):
        temp_i = sort_index[num]
        file_path = os.path.join(load_csv_name_path, file_list[temp_i])
        temp_data = pd.read_csv(file_path)
        temp_data.to_csv(os.path.join(save_csv_name_path, str(num) + '_retrain.csv'), index=False)
        # print(len(temp_data))

def select_all_bucket(path, file_list):
    flag = 0
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        temp_data = pd.read_csv(file_path)
        if flag == 0:
            current_data = temp_data
        else:
            current_data = pd.concat([current_data, temp_data], ignore_index=True)
    return current_data


def get_coverage_data(dataset_name, model_name, save_path):
    coverage_path = os.path.join(save_path, model_name, dataset_name, 'sort_coverage')
    file_list = os.listdir(coverage_path)
    bucket_num = set_coverage_bucket(dataset_name)
    all_bucket_num = len(file_list)
    coverage_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for rate_num in range(len(coverage_rate_list)):
        temp_rate = coverage_rate_list[rate_num]
        print('rate:' + str(temp_rate))
        max_bucket_num = int(temp_rate * bucket_num)
        if all_bucket_num <= max_bucket_num:
            current_data = select_all_bucket(coverage_path, file_list)
        else:
            temp_bucket = 0
            start_num = 0
            while temp_bucket < max_bucket_num:
                bucket_data_path = os.path.join(coverage_path, str(start_num) + '_retrain.csv')
                temp_data = pd.read_csv(bucket_data_path)

                if temp_bucket == 0:
                    current_data = temp_data
                else:
                    current_data = pd.concat([current_data, temp_data], ignore_index=True)
                temp_bucket += 1
                if temp_bucket >= max_bucket_num:
                    break
                bucket_data_path = os.path.join(coverage_path, str(all_bucket_num-start_num-1) + '_retrain.csv')
                temp_data = pd.read_csv(bucket_data_path)
                current_data = pd.concat([current_data, temp_data], ignore_index=True)
                temp_bucket += 1
                start_num += 1
        coverage_data_path = os.path.join(save_path, model_name, dataset_name, 'coverage_data')
        if not os.path.exists(coverage_data_path):
            os.makedirs(coverage_data_path)
        current_data.to_csv(os.path.join(coverage_data_path, str(temp_rate) + '_retrain.csv'), index=False)
        print('num:' + str(len(current_data)))

def show_bucket_num(dataset_name, model_name, load_path):

    bucket_num = set_coverage_bucket(dataset_name)

    values = np.load(os.path.join(load_path, model_name, dataset_name, 'lime_scores.npy'))

    bucket_arr = np.zeros(bucket_num + 1, dtype='int')
    interval = 1.0 / float(bucket_num)
    for values_num in range(len(values)):
        temp_value = values[values_num]
        # print(temp_value)
        index = int(temp_value / interval)
        # print(index)
        bucket_arr[int(index)] += 1
        # print(temp_value)
    # print(bucket_arr)
    times = np.count_nonzero(bucket_arr)
    print("total bucket num:" + str(bucket_num))
    print("coverage bucket num:" + str(times))
    coverage = times / float(bucket_num + 1)
    print("coverage bucket num:" + str(coverage))

if __name__ == '__main__':
    pass







