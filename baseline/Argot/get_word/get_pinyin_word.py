from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultDagParams, dag, simplify_pinyin


def simplify_py(pinyin_list):
    simplify_pinyin_list = []
    for i in range(len(pinyin_list)):
        temp = pinyin_list[i]
        temp = simplify_pinyin(temp)
        simplify_pinyin_list.append(temp)
    return simplify_pinyin_list

def replace_an_and_ang(string):
    str1 = 'an'
    str2 = 'ang'
    position1 = -1
    position1 = string.find(str1, position1 + 1)
    position2 = -1
    position2 = string.find(str2, position2 + 1)
    if position2 != -1:
        string = string.replace(str2, str1)
    elif position1 != -1:
        string = string.replace(str1, str2)
    return string

def replace_en_and_eng(string):
    str1 = 'en'
    str2 = 'eng'
    position1 = -1
    position1 = string.find(str1, position1 + 1)
    position2 = -1
    position2 = string.find(str2, position2 + 1)
    if position2 != -1:
        string = string.replace(str2, str1)
    elif position1 != -1:
        string = string.replace(str1, str2)
    return string

def replace_in_and_ing(string):
    str1 = 'in'
    str2 = 'ing'
    position1 = -1
    position1 = string.find(str1, position1 + 1)
    position2 = -1
    position2 = string.find(str2, position2 + 1)
    if position2 != -1:
        string = string.replace(str2, str1)
    elif position1 != -1:
        string = string.replace(str1, str2)
    return string

def replace_c_and_ch(string):
    str1 = 'c'
    str2 = 'ch'
    position1 = -1
    position1 = string.find(str1, position1 + 1)
    position2 = -1
    position2 = string.find(str2, position2 + 1)
    if position2 != -1:
        string = string.replace(str2, str1)
    elif position1 != -1:
        string = string.replace(str1, str2)
    return string

def replace_z_and_zh(string):
    str1 = 'z'
    str2 = 'zh'
    position1 = -1
    position1 = string.find(str1, position1 + 1)
    position2 = -1
    position2 = string.find(str2, position2 + 1)
    if position2 != -1:
        string = string.replace(str2, str1)
    elif position1 != -1:
        string = string.replace(str1, str2)
    return string

def replace_s_and_sh(string):
    str1 = 's'
    str2 = 'sh'
    position1 = -1
    position1 = string.find(str1, position1 + 1)
    position2 = -1
    position2 = string.find(str2, position2 + 1)
    if position2 != -1:
        string = string.replace(str2, str1)
    elif position1 != -1:
        string = string.replace(str1, str2)
    return string

def head_and_tail_nasal_replace(pinyin_list):
    temp_pinyin_list = []
    for i in range(len(pinyin_list)):
        temp = pinyin_list[i]
        temp = replace_an_and_ang(temp)
        temp = replace_in_and_ing(temp)
        temp = replace_en_and_eng(temp)
        temp_pinyin_list.append(temp)
    return temp_pinyin_list

def flat_tongues_replace(pinyin_list):
    temp_pinyin_list = []
    for i in range(len(pinyin_list)):
        temp = pinyin_list[i]
        temp = replace_z_and_zh(temp)
        temp = replace_c_and_ch(temp)
        temp = replace_s_and_sh(temp)
        temp_pinyin_list.append(temp)
    return temp_pinyin_list

def pinyin_2_hanzi(pinyin_list, str, max_num=10):
    pinyin_list = simplify_py(pinyin_list)
    # print(pinyin_list)
    dagParams = DefaultDagParams()
    result = dag(dagParams, pinyin_list, path_num=max_num, log=True)
    max_num = len(result)
    if max_num == 0:
        return str
    for i in range(max_num):
        temp_string = ''
        string_list = result[i].path
        for num in range(len(string_list)):
            temp_string += string_list[num]
        if temp_string != str:
            break
    return temp_string

def get_pinyin_word(seg):
    # print(seg)
    final_string = ''
    temp_string = ''
    i = 0
    while i < len(seg):
        temp_str = seg[i]
        if '\u4e00' <= temp_str <= '\u9fff':
            temp_string += temp_str
        else:
            if len(temp_string) != 0:
                # print(temp_string)
                temp_pinyin = lazy_pinyin(temp_string)
                temp_pinyin = head_and_tail_nasal_replace(temp_pinyin)
                temp_pinyin = flat_tongues_replace(temp_pinyin)
                temp_result = pinyin_2_hanzi(temp_pinyin, temp_string)
                final_string += temp_result
                temp_string = ''
            final_string += temp_str
        i += 1
    if len(temp_string) != 0:
        temp_pinyin = lazy_pinyin(temp_string)
        temp_pinyin = head_and_tail_nasal_replace(temp_pinyin)
        temp_pinyin = flat_tongues_replace(temp_pinyin)
        temp_result = pinyin_2_hanzi(temp_pinyin, temp_string)
        final_string += temp_result
    # print(final_string)
    return final_string

if __name__ == '__main__':
   pass




