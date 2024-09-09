import numpy as np
import os

def find_character(character, characters):
    characters = list(characters)
    try:
        position = characters.index(character)
        return position
    except ValueError:
        return -1

def get_del_radical_character(temp_str):
    characters = np.load(os.path.join('/media/usr/external/home/usr/project/project2_data/baseline/Argot/chaizi', 'characters.npy'), allow_pickle=False)
    splitting_words = np.load(os.path.join('/media/usr/external/home/usr/project/project2_data/baseline/Argot/chaizi', 'splitting_words.npy'), allow_pickle=False)
    position = find_character(temp_str, characters)
    similar_characters = []
    if position == -1:
        return similar_characters
    splitting_word_list = splitting_words[position]
    for num in range(len(splitting_word_list)):
        if splitting_word_list[num] == ' ':
            continue
        similar_characters.append(splitting_word_list[num])
    return similar_characters

if __name__ == '__main__':
    pass


