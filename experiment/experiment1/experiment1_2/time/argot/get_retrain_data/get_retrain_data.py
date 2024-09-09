import os
import pandas as pd
from experiment.experiment2.experiment2_3.set_retrain_data_num import set_retrain_data_num

def get_retrain_data(dataset_name, model_name, load_step_path, save_path):
    important_num_list = [1, 2, 12]
    if dataset_name == 'weibo':
        important_num = important_num_list[0]
    elif dataset_name == 'news':
        important_num = important_num_list[1]
    else:
        important_num = important_num_list[2]
    load_add_path = os.path.join(load_step_path, str(important_num), model_name, dataset_name)
    add_data = pd.read_csv(os.path.join(load_add_path, 'step1_final_mutants.csv'))
    text_list = []
    label_list = []
    for add_data_num in range(len(add_data)):
        text_list.append(add_data.loc[add_data_num, 'mutant'])
        label_list.append(add_data.loc[add_data_num, 'label'])
    merge_dt_dict = {'text': text_list, 'label': label_list}
    add_data = pd.DataFrame(merge_dt_dict)
    add_data.to_csv(os.path.join(save_path, model_name, dataset_name, 'add.csv'), index=False)
    print(len(add_data))
    train_data = pd.read_csv(os.path.join(save_path, model_name, dataset_name, 'train.csv'))
    retrain_data = pd.concat([train_data, add_data], ignore_index=True)
    retrain_data.to_csv(os.path.join(save_path, model_name, dataset_name, 'train.csv'), index=False)
    print(len(train_data))
    print(len(retrain_data))

if __name__ == '__main__':

    pass
