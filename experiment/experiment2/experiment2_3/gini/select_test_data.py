import numpy as np
import os
import pandas as pd
import jieba

from experiment.experiment2.experiment2_3.set_retrain_data_num import set_retrain_data_num


def select_test_data(labels, dataset_name, model_name, load_step_path):
    load_value_path = os.path.join(load_step_path, model_name, dataset_name)
    test_data = pd.read_csv(os.path.join(load_value_path, 'test.csv'))
    gini_value = test_data['DeepGini_scores']
    gini_value = np.array(gini_value)
    sort_index = np.argsort(gini_value)[::-1]
    data_num = set_retrain_data_num(dataset_name)
    # print(data_num)
    class_data_num = int(data_num / len(labels))
    # print(class_data_num)
    text_list = []
    label_list = []

    for class_num in range(len(labels)):
        temp_num = 0
        for index_num in range(len(sort_index)):
            temp_index = sort_index[index_num]
            if test_data.loc[temp_index, 'label'] != class_num:
                continue
            text_list.append(test_data.loc[temp_index, 'text'])
            label_list.append(test_data.loc[temp_index, 'label'])
            temp_num += 1
            if temp_num == class_data_num:
                break

    merge_dt_dict = {'text': text_list, 'label': label_list}
    data_df = pd.DataFrame(merge_dt_dict)
    data_df.to_csv(os.path.join(load_value_path, 'add.csv'), index=False)

    train_data = pd.read_csv(os.path.join(load_value_path, 'train.csv'))
    retrain_data = pd.concat([train_data, data_df], ignore_index=True)
    retrain_data.to_csv(os.path.join(load_value_path, 'train.csv'), index=False)
    print(len(retrain_data))


if __name__ == '__main__':

    pass