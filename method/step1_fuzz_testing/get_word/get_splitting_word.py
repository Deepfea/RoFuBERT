import random
import time
from pypinyin import lazy_pinyin
from Pinyin2Hanzi import  DefaultDagParams, dag, simplify_pinyin
import torch
import numpy as np
import os

def get_all_word():
    path = '/media/usr/external/home/usr/project/project2_data/chaizi/chaizi.txt'
    character_arr = []
    splitting_word = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            string = line.strip()
            print(len(string))
            split_str = string.split("	")
            print(split_str)
            character_arr.append(split_str[0])
            splitting_word.append(split_str[1])
    character_arr = np.array(character_arr)
    splitting_word = np.array(splitting_word)
    np.save(os.path.join('/media/usr/external/home/usr/project/project2_data/chaizi', 'characters.npy'), character_arr)
    np.save(os.path.join('/media/usr/external/home/usr/project/project2_data/chaizi', 'splitting_words.npy'), splitting_word)


def find_character(character, characters):
    characters = list(characters)
    try:
        position = characters.index(character)
        return position
    except ValueError:
        return -1

def get_word(splitting_words, position, temp_str):
    if position == -1:
        return temp_str
    else:
        words = ''
        words_list = splitting_words[position]
        for num in range(len(words_list)):
            words += words_list[num]
        return words

def get_splitting_word(temp_seg):
    characters = np.load(os.path.join('/media/usr/external/home/usr/project/project2_data/chaizi', 'characters.npy'), allow_pickle=False)
    splitting_words = np.load(os.path.join('/media/usr/external/home/usr/project/project2_data/chaizi', 'splitting_words.npy'), allow_pickle=False)
    # print(len(characters))
    # print(len(splitting_words))
    final_string = ''
    i = 0
    while i < len(temp_seg):
        temp_str = temp_seg[i]
        if '\u4e00' <= temp_str <= '\u9fff':
            # print(temp_str)
            position = find_character(temp_str, characters)
            words = get_word(splitting_words, position, temp_str)
            final_string += words
        else:
            final_string += temp_str
        i += 1
    final_string = final_string.replace(' ', '')
    return final_string

if __name__ == '__main__':
   pass


