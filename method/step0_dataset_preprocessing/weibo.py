import os
import pandas as pd
from xml.etree import cElementTree as ET
from collections import Counter

base_path = '/media/usr/external/home/usr/project/project2_data/dataset/weibo'
# name_list = os.listdir(base_path)
# label_list = []
# text_list = []
# for file_name in name_list:
#     file_path = os.path.join(base_path, file_name)
#     tree = ET.parse(file_path)
#     root = tree.getroot()
#     xml_str = ET.tostring(root)
#     root = ET.fromstring(xml_str)
#     for sentences in root:
#         for sentence in sentences:
#             # print(sentence.attrib)
#             if sentence.attrib['opinionated'] == 'N':
#                 emotion = 'none'
#             else:
#                 emotion = sentence.attrib['emotion-1-type']
#             text = sentence.text
#             label_list.append(emotion)
#             text_list.append(text)
# data = {
#     'text': text_list,
#     'emotion': label_list
#         }
# all_df = pd.DataFrame(data)
# values_list = all_df['emotion'].tolist()
# counts = Counter(values_list)
# print(counts)
#
# # 统计整个数据集已有的不同类别的数据量，方便做成我们要的数据集
#
labels = ['none', 'like', 'disgust', 'happiness', 'sadness']
# for label_i in range(len(labels)):
#     label = []
#     label.append(labels[label_i])
#     filtered_df = all_df[all_df['emotion'].isin(label)]
#     filtered_df = filtered_df.reset_index(drop=True)
#     values_list = filtered_df['emotion'].tolist()
#     # 使用Counter统计各元素的数量
#     counts = Counter(values_list)
#     print(counts)
#     temp_list = []
#     for num in range(len(filtered_df)):
#         temp_list.append(label_i)
#     filtered_df['label'] = temp_list
#
#     filtered_df.to_csv(os.path.join(base_path, str(label_i) + '.csv'), index=False)

# 生成并保存训练集、验证集和测试集
for label_i in range(len(labels)):
    temp_df = pd.read_csv(os.path.join(base_path, str(label_i) + '.csv'))
    temp_df = temp_df.sample(frac=1).reset_index(drop=True)
    # print(len(temp_df))
    # temp_df1 = temp_df[:2250]
    # temp_df2 = temp_df[2250:2500]
    # temp_df3 = temp_df[2500:2750]
    temp_df1 = temp_df[:2000]
    temp_df2 = temp_df[2000:2250]
    temp_df3 = temp_df[2250:2500]
    if label_i == 0:
        train_df = temp_df1
        val_df = temp_df2
        test_df = temp_df3
    else:
        train_df = pd.concat([train_df, temp_df1], ignore_index=True)
        val_df = pd.concat([val_df, temp_df2], ignore_index=True)
        test_df = pd.concat([test_df, temp_df3], ignore_index=True)
# print(train_df)
# print(val_df)
# print(test_df)

for num in range(len(train_df)):
    temp_text = train_df.loc[num, 'text']
    temp_text = temp_text.replace(' ', '')
    temp_text = temp_text.replace('　', '')
    # print(temp_text)
    if '\u4e00' <= temp_text[len(temp_text)-1] <= '\u9fff':
        temp_text += '。'
    train_df.loc[num, 'text'] = temp_text

for num in range(len(val_df)):
    temp_text = val_df.loc[num, 'text']
    temp_text = temp_text.replace(' ', '')
    temp_text = temp_text.replace('　', '')
    # print(temp_text)
    if '\u4e00' <= temp_text[len(temp_text)-1] <= '\u9fff':
        temp_text += '。'
    val_df.loc[num, 'text'] = temp_text

for num in range(len(test_df)):
    temp_text = test_df.loc[num, 'text']
    # print(temp_text)
    temp_text = temp_text.replace(' ', '')
    temp_text = temp_text.replace('　', '')
    # print(temp_text)
    if '\u4e00' <= temp_text[len(temp_text)-1] <= '\u9fff':
        temp_text += '。'
    test_df.loc[num, 'text'] = temp_text
    # print(temp_text)

print(train_df)
print(val_df)
print(test_df)

train_df.to_csv(os.path.join(base_path, 'train.csv'), index=False)
val_df.to_csv(os.path.join(base_path, 'val.csv'), index=False)
test_df.to_csv(os.path.join(base_path, 'test.csv'), index=False)