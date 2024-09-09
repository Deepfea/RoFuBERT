import os
import pandas as pd
from experiment.experiment2.experiment2_3.set_retrain_data_num import set_retrain_data_num

def count_modify_rate(dataset_name, model_name, load_step_path):
    important_num_list = [1, 2, 12]
    if dataset_name == 'weibo':
        important_num = important_num_list[0]
    elif dataset_name == 'news':
        important_num = important_num_list[1]
    else:
        important_num = important_num_list[2]
    load_add_path = os.path.join(load_step_path, 'Argot', str(important_num), model_name, dataset_name)
    add_data = pd.read_csv(os.path.join(load_add_path, 'step1_final_mutants.csv'))
    avg_modify_rate = 0
    for add_data_num in range(len(add_data)):
        temp_word = add_data.loc[add_data_num, 'word']
        temp_word = temp_word.replace("+", "")
        temp_mutant = add_data.loc[add_data_num, 'mutant']
        temp_rate = float(len(temp_word)) / float(len(temp_mutant))
        avg_modify_rate += temp_rate
        # print(temp_rate)
    avg_modify_rate = avg_modify_rate / float(len(add_data))
    print(avg_modify_rate)

if __name__ == '__main__':

    pass
