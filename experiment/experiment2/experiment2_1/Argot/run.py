import os

from baseline.Argot.step0_cal_test_output import cal_output
from baseline.Argot.step1_cut_seg import cut_seg
from baseline.Argot.step2_cal_word_scores import cal_word_scores, select_segs
from baseline.Argot.step3_generate_mutants import generate_mutant1
from baseline.Argot.step4_select_mutants import select_mutants


def generate_adv(dataset_name, model_name, labels, dataset_path, train_path, save_path):
    print(dataset_name)
    print(model_name)
    important_num_list = [12, 2, 1]
    if dataset_name == 'weibo':
        important_num = important_num_list[2]
    elif dataset_name == 'news':
        important_num = important_num_list[1]
    else:
        important_num = important_num_list[0]
    print(important_num)
    # cal_output(dataset_name, model_name, dataset_path, train_path, save_path)
    # cut_seg(labels, dataset_name, model_name, save_path)
    # cal_word_scores(labels, dataset_name, model_name, train_path, save_path)
    save_path_1 = os.path.join(save_path, str(important_num))
    select_segs(dataset_name, model_name, save_path, save_path_1, important_num)
    generate_mutant1(dataset_name, model_name, save_path, save_path_1)
    select_mutants(dataset_name, model_name, train_path, save_path, save_path_1)

if __name__ == '__main__':
    pass
