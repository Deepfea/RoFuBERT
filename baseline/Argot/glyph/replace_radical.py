import numpy as np
import os

def find_character(character, characters):
    characters = list(characters)
    try:
        position = characters.index(character)
        return position
    except ValueError:
        return -1

def get_radical_replace_character(temp_str, max_num=10):
    characters = np.load(os.path.join('/media/usr/external/home/usr/project/project2_data/baseline/Argot/chaizi', 'characters.npy'), allow_pickle=False)
    similar_characters = []
    position = find_character(temp_str, characters)
    if position == -1:
        return similar_characters
    else:
        num = position
        if int(num - max_num / 2) < 0:
            num -= 1
            while num >= 0:
                similar_characters.append(characters[num])
                num -= 1
                max_num -= 1
            num = position + 1
            while max_num > 0:
                similar_characters.append(characters[num])
                num += 1
                max_num -= 1
        elif int(num + max_num / 2) >= len(characters):
            num += 1
            while num < len(characters):
                similar_characters.append(characters[num])
                num += 1
                max_num -= 1
            num = position - 1
            while max_num > 0:
                similar_characters.append(characters[num])
                num -= 1
                max_num -= 1
        else:
            for i in range(int(max_num / 2)):
                similar_characters.append(characters[num + i + 1])
                similar_characters.append(characters[num - i - 1])
        return similar_characters

if __name__ == '__main__':
    pass


