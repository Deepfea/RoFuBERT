import numpy as np
from PIL import Image, ImageDraw, ImageFont
from baseline.Argot.glyph.model.siamese import Siamese

def convert_pic(charater):

    # 创建一个新的白色图片
    width, height = 105, 105
    image = Image.new('RGB', (width, height), color='white')
    # 设置字体
    font = ImageFont.truetype("/media/usr/external/home/usr/project/project2_data/baseline/Argot/xingjinzi/simsun.ttc", 85)
    # 绘制文本
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), charater, font=font, fill=(0, 0, 0))
    return image

def predict(str1, str2_list):
    # print(str1)
    # print(str2_list)
    model = Siamese()
    image1 = convert_pic(str1)
    output_list = []
    for num in range(len(str2_list)):
        image2 = convert_pic(str2_list[num])
        probability = model.detect_image(image1, image2)
        output = probability.cpu().numpy()[0]
        output_list.append(output)
    return output_list


if __name__ == "__main__":
    model = Siamese()
        
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        probability = model.detect_image(image_1,image_2)
        print(probability)
