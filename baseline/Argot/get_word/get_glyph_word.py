import os

import numpy as np

from baseline.Argot.glyph.delete_radical import get_del_radical_character
from baseline.Argot.glyph.model.step2_predict import predict
from baseline.Argot.glyph.replace_radical import get_radical_replace_character

def get_glyph_word(temp_seg):
    final_string = ''
    i = 0
    while i < len(temp_seg):
        temp_str = temp_seg[i]
        similar_str = []
        # print(temp_str)
        if '\u4e00' <= temp_str <= '\u9fff':
            temp_list = get_radical_replace_character(temp_str, max_num=4)
            similar_str += temp_list
            temp_list = get_del_radical_character(temp_str)
            similar_str += temp_list
            temp_str = choose_str(temp_str, similar_str)
        final_string += temp_str
        i += 1
    return final_string

def choose_str(temp_str, similar_str):
    x = temp_str
    scores_list = predict(x, similar_str)
    # print(scores_list)
    if len(scores_list) == 0:
        return ''
    scores = np.array(scores_list)
    index = np.argmax(scores)
    final_str = similar_str[index]
    return final_str

def find_all_occurrences(lst, value, characters):
    indices = [index for index, item in enumerate(lst) if item == value]
    character_list = []
    for i in range(len(indices)):
        character_list.append(characters[indices[i]])
    return character_list

if __name__ == '__main__':
   pass