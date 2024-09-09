import os
import pandas as pd
from experiment.experiment2.experiment2_3.set_retrain_data_num import set_retrain_data_num


def get_retrain_data(labels, dataset_name, model_name, load_step_path, important_num):
    load_test_path = os.path.join(load_step_path, model_name, dataset_name)
    load_mutant_path = os.path.join(load_test_path, str(important_num))
    data_num = set_retrain_data_num(dataset_name)
    add2_data = pd.read_csv(os.path.join(load_mutant_path, 'final_mutants.csv'))
    text_list = []
    label_list = []
    for add_data_num in range(len(add2_data)):
        text_list.append(add2_data.loc[add_data_num, 'mutant'])
        label_list.append(add2_data.loc[add_data_num, 'label'])
        if len(text_list) == data_num:
                break
    merge_dt_dict = {'text': text_list, 'label': label_list}
    adv_data = pd.DataFrame(merge_dt_dict)
    train_data = pd.read_csv(os.path.join(load_test_path, 'train.csv'))
    retrain_data = pd.concat([train_data, adv_data], ignore_index=True)
    retrain_data.to_csv(os.path.join(load_test_path, 'train.csv'), index=False)
    print(len(adv_data))
    print(len(train_data))
    print(len(retrain_data))

if __name__ == '__main__':
    pass

