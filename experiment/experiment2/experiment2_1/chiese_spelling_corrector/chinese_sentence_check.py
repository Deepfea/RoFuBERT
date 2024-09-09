import os.path
import numpy as np
import pandas as pd
from pycorrector import MacBertCorrector
import os
def rofubert_chinese_sentence_check(dataset_name, model_name, dataset_path, load_path, save_path):
    if dataset_name == 'weibo' or dataset_name == 'cail':
        max_length = 256
    else:
        max_length = 64

    print('rofubert:')
    print(dataset_name)
    print(model_name)
    save_path = os.path.join(save_path, model_name, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    load_test_path = os.path.join(dataset_path, 'test.csv')
    test_data = pd.read_csv(load_test_path)
    texts = test_data['text']
    load_add_path = os.path.join(load_path, model_name, dataset_name, 'add.csv')
    add_data = pd.read_csv(load_add_path)
    text_index = add_data['origin']
    mutants = add_data['mutant']
    model = MacBertCorrector('/media/usr/external/home/usr/project/project2_data/epxeriment/experiment2/experiment2_1/chinese_spelling_corrector/model')
    result = model.correct_batch(mutants, max_length=max_length)
    mutant_num = len(mutants)
    total_error_num = 0
    for num in range(len(result)):
        total_error_num += len(result[num]['errors'])
    avg_error_num = total_error_num / float(mutant_num)
    print('对抗样本数量:' + str(mutant_num))
    print('总错误数量:' + str(total_error_num))
    print('平均错误数量:' + str(avg_error_num))
    save_list = []
    save_list.append(mutant_num)
    save_list.append(total_error_num)
    save_list.append(avg_error_num)
    save_arr = np.array(save_list)
    np.save(os.path.join(save_path, 'rofubert_chinese_sentence_check_results.npy'), save_arr)

def argot_chinese_sentence_check(dataset_name, model_name, load_path, save_path):
    print('argot:')
    print(dataset_name)
    print(model_name)
    important_num_list = [12, 2, 1]
    if dataset_name == 'weibo':
        important_num = important_num_list[2]
        max_length = 256
    elif dataset_name == 'news':
        important_num = important_num_list[1]
        max_length = 64
    else:
        important_num = important_num_list[0]
        max_length = 256
    save_path = os.path.join(save_path, model_name, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    load_mutant_path = os.path.join(load_path, str(important_num), model_name, dataset_name, 'step1_final_mutants.csv')
    data = pd.read_csv(load_mutant_path)
    texts = data['text']
    mutants = data['mutant']

    model = MacBertCorrector('/media/usr/external/home/usr/project/project2_data/epxeriment/experiment2/experiment2_1/chinese_spelling_corrector/model')
    result = model.correct_batch(mutants, max_length=max_length)
    mutant_num = len(mutants)
    total_error_num = 0
    for num in range(len(result)):
        total_error_num += len(result[num]['errors'])
        # if len(result[num]['errors']) > 6:
        #     print(texts[num])
        #     print(mutants[num])
        #     print(result[num]['errors'])
    avg_error_num = total_error_num / float(mutant_num)
    print('对抗样本数量:' + str(mutant_num))
    print('总错误数量:' + str(total_error_num))
    print('平均错误数量:' + str(avg_error_num))
    save_list = []
    save_list.append(mutant_num)
    save_list.append(total_error_num)
    save_list.append(avg_error_num)
    save_arr = np.array(save_list)
    np.save(os.path.join(save_path, 'argot_chinese_sentence_check_results.npy'), save_arr)

if __name__ == '__main__':
    pass
