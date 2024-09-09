import numpy as np
import os
import pandas as pd
import jieba

from experiment.experiment2.experiment2_3.set_retrain_data_num import set_retrain_data_num


def select_test_data(labels, dataset_name, model_name, load_step_path):
    load_value_path = os.path.join(load_step_path, model_name, dataset_name)
    seed_data = pd.read_csv(os.path.join(load_value_path, 'seed.csv'))
    seed_outputs = np.load(os.path.join(load_value_path, 'seed_outputs.npy'))
    gini_value = np.load(os.path.join(load_value_path, 'seed_gini_scores.npy'))
    sort_index = np.argsort(gini_value)[::-1]
    if dataset_name == 'weibo':
        data_num = set_retrain_data_num(dataset_name) - (1250 - len(gini_value))
        print(1250 - len(gini_value))
    else:
        data_num = set_retrain_data_num(dataset_name) - (5000 - len(gini_value))
        print(5000 - len(gini_value))

    # print(data_num)

    text_list = []
    label_list = []
    outputs_list = []
    temp_num = 0
    for index_num in range(data_num):
        temp_index = sort_index[index_num]
        text_list.append(seed_data.loc[temp_index, 'text'])
        label_list.append(seed_data.loc[temp_index, 'label'])
        outputs_list.append(seed_outputs[temp_index])
        temp_num += 1
    merge_dt_dict = {'text': text_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    data_df.to_csv(os.path.join(load_value_path, 'final_seed.csv'), index=False)
    print(len(data_df))
    outputs_arr = np.array(outputs_list)
    np.save(os.path.join(load_value_path, 'final_seed_outputs.npy'), outputs_arr)
    print(outputs_arr.shape)


if __name__ == '__main__':

    pass