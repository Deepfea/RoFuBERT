import os
import pandas as pd
from torch import nn
import time
from method.step1_fuzz_testing.step0_select_seeds.cal_deepgini_value import cal_deepGini
from method.step1_fuzz_testing.step0_select_seeds.select_seeds import select_seed_and_segmentation
from method.step1_fuzz_testing.step1_generate_mutants.cal_word_scores import cal_word_scores
from method.step1_fuzz_testing.step1_generate_mutants.generate_mutants import generate_mutant1
from method.step1_fuzz_testing.step2_cal_coverage_rate.cal_coverage import cal_coverage, cal_add_coverage
from method.step1_fuzz_testing.step2_cal_coverage_rate.cal_lime_values import cal_mutant_lime_value, cal_ori_lime_value
from method.step1_fuzz_testing.step3_renew_test_data.renew_testing_data import del_seed_in_testing

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, len(labels))
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

def load_test_len(dataset_name, model_name, load_step1_path):
    load_test_path = os.path.join(load_step1_path, model_name, dataset_name)
    test_data = pd.read_csv(os.path.join(load_test_path, 'step0_test.csv'))
    return len(test_data)

def fuzz_testing(dataset_name, model_name, labels, dataset_path, load_step0_path, load_step1_path, important_num):
    print(dataset_name)
    print(model_name)
    counter = 0

    # # # 计算训练集的lime值
    # print('初始化：计算训练集的lime值')
    # cal_ori_lime_value(dataset_name, model_name, labels, dataset_path, load_step0_path, load_step1_path)
    # print('初始化：计算训练集的lime值完成')

    # 计算训练集的lime值的数组
    print('初始化：计算训练集的lime值覆盖率')
    cal_coverage(load_step1_path, model_name, dataset_name)
    print('初始化：计算训练集的lime值覆盖率完成')

    # 计算所有测试集的deepgini值
    print('初始化：计算所有测试集的deepgini值')
    cal_deepGini(dataset_name, model_name, dataset_path, load_step0_path, load_step1_path, counter)
    print('初始化：计算所有测试集的deepgini值完成')

    test_data_num = load_test_len(dataset_name, model_name, load_step1_path)

    while test_data_num != 0:
        counter += 1
        print('当前种子数量剩余：' + str(test_data_num))

        # 选择种子并得到不同的词组
        print('选择种子并得到不同的词组')
        select_seed_and_segmentation(labels, dataset_name, model_name, load_step1_path)
        print('选择种子并得到不同的词组完成')

        # 从词组中筛选得到重要词组
        print('从词组中筛选得到重要词组')
        cal_word_scores(labels, dataset_name, model_name, load_step0_path, load_step1_path, important_num)
        print('从词组中筛选得到重要词组完成')

        # 根据重要词组生成句子
        print('根据重要词组生成句子')
        generate_mutant1(labels, dataset_name, model_name, load_step0_path, load_step1_path, counter)
        print('根据重要词组生成句子完成')

        # 句子得到不同的得分
        print('句子得到不同的得分')
        cal_mutant_lime_value(dataset_name, model_name, labels, load_step0_path, load_step1_path)
        print('句子得到不同的得分完成')

        # 根据已有的覆盖率选择可以增长的句子并更新覆盖率
        print('根据已有的覆盖率选择可以增长的句子并更新覆盖率')
        cal_add_coverage(load_step1_path, model_name, dataset_name, counter)
        print('根据已有的覆盖率选择可以增长的句子并更新覆盖率完成')

        # 更新测试集
        print('更新测试集')
        del_seed_in_testing(dataset_name, model_name, load_step1_path)
        print('更新测试集完成')

        # 更新测试集的数量
        test_data_num = load_test_len(dataset_name, model_name, load_step1_path)

if __name__ == '__main__':
    pass






