import jieba
import numpy as np
import math

def get_seg(str):
    cut_result = jieba.cut(str, cut_all=False)
    cut_result = list(cut_result)
    # print(cut_result)
    return cut_result

def get_cos_value(temp_seg_list1, temp_seg_list2):
    temp_seg_list = list(set(temp_seg_list1 + temp_seg_list2))
    # print(temp_seg_list)
    word_fre1 = np.zeros(len(temp_seg_list), dtype='int')
    word_fre2 = np.zeros(len(temp_seg_list), dtype='int')
    for num in range(len(temp_seg_list)):
        element = temp_seg_list[num]
        word_fre1[num] = temp_seg_list1.count(element)
        word_fre2[num] = temp_seg_list2.count(element)
    # print(word_fre1)
    # print(word_fre2)
    x = 0
    y = 0
    z = 0
    for num in range(len(word_fre1)):
        x += word_fre1[num] * word_fre2[num]
        y += word_fre1[num] * word_fre1[num]
        z += word_fre2[num] * word_fre2[num]
    y = math.sqrt(y)
    z = math.sqrt(z)
    result = x / float(y * z)
    return result

def get_sim_value(str1, str2):
    temp_seg_list1 = get_seg(str1)
    temp_seg_list2 = get_seg(str2)
    value = get_cos_value(temp_seg_list1, temp_seg_list2)

    return value

if __name__ == '__main__':
    pass
