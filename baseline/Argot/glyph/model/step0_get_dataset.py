from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os

def convert_pic(charater, character_path):

    # 创建一个新的白色图片
    width, height = 105, 105
    image = Image.new('RGB', (width, height), color='white')
    # 设置字体
    font = ImageFont.truetype("/media/usr/external/home/usr/project/project2_data/baseline/xingjinzi/simsun.ttc", 85)

    # 绘制文本
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), charater, font=font, fill=(0, 0, 0))

    # 展示图片
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    image.save(character_path)

def get_character(path):
    characters = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            similar_characters = []
            # 移除换行符，并且根据空格拆分
            splits = line.strip('\n')
            for num in range(len(splits)):
                similar_characters.append(splits[num])
            characters.append(similar_characters)
    return characters

def get_pic(characters, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 创建文件夹
    num = 0
    while num < len(characters):
        save_class_path = os.path.join(save_path, 'charachers' + str(num))
        if not os.path.exists(save_class_path):
            os.makedirs(save_class_path)
        temp_characters = characters[num]
        # print(temp_characters)
        temp_num = 0
        while temp_num < len(temp_characters):
            temp_character = temp_characters[temp_num]
            character_path = os.path.join(save_class_path, str(num) + '_' + str(temp_num) + '.png')
            convert_pic(temp_character, character_path)
            temp_num += 1
        num += 1



if __name__ == "__main__":
    pass


