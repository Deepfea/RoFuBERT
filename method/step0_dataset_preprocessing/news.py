import os
import pandas as pd
from collections import Counter

base_path = '/media/usr/external/home/usr/project/project2_data/dataset/THUCNews'
# classes_list = os.listdir(base_path)
# label_list = []
# text_list = []
# class_list = []
# for class_name in classes_list:
#     print(class_name)
#     class_list.append(class_name)
#     class_path = os.path.join(base_path, class_name)
#     file_list = os.listdir(class_path)
#     for file_name in file_list:
#         file_path = os.path.join(class_path, file_name)
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#             first_line = lines[0]
#             first_line = first_line.split('\n')[0]
#             text_list.append(first_line)
#             label_list.append(class_name)
#         file.close()
# data = {
#     'text': text_list,
#     'class': label_list
#         }
# all_df = pd.DataFrame(data)
# values_list = all_df['class'].tolist()
# counts = Counter(values_list)
# print(counts)
labels = ['家居', '股票', '娱乐', '游戏', '社会', '科技', '时政', '体育', '教育', '财经']
# for label_i in range(len(labels)):
#     label = []
#     label.append(labels[label_i])
#     filtered_df = all_df[all_df['class'].isin(label)]
#     filtered_df = filtered_df.reset_index(drop=True)
#     values_list = filtered_df['class'].tolist()
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
    temp_df1 = temp_df[:4000]
    temp_df2 = temp_df[4000:4500]
    temp_df3 = temp_df[4500:5000]
    if label_i == 0:
        train_df = temp_df1
        val_df = temp_df2
        test_df = temp_df3
    else:
        train_df = pd.concat([train_df, temp_df1], ignore_index=True)
        val_df = pd.concat([val_df, temp_df2], ignore_index=True)
        test_df = pd.concat([test_df, temp_df3], ignore_index=True)

temp_str1 = '，'
temp_str2 = '。'
for num in range(len(train_df)):
    temp_text = train_df.loc[num, 'text']
    temp_text = temp_text.replace(' ', temp_str1)
    # print(temp_text)
    while temp_text.find('，，') >= 0:
        temp_text = temp_text.replace('，，', temp_str1)
    temp_text += temp_str2
    temp_text = temp_text.replace('，。', temp_str2)
    train_df.loc[num, 'text'] = temp_text

for num in range(len(val_df)):
    temp_text = val_df.loc[num, 'text']
    temp_text = temp_text.replace(' ', temp_str1)
    # print(temp_text)
    while temp_text.find('，，') >= 0:
        temp_text = temp_text.replace('，，', temp_str1)
    temp_text += temp_str2
    temp_text = temp_text.replace('，。', temp_str2)
    val_df.loc[num, 'text'] = temp_text

for num in range(len(test_df)):
    temp_text = test_df.loc[num, 'text']
    temp_text = temp_text.replace(' ', temp_str1)
    # print(temp_text)
    while temp_text.find('，，') >= 0:
        temp_text = temp_text.replace('，，', temp_str1)
    temp_text += temp_str2
    temp_text = temp_text.replace('，。', temp_str2)
    test_df.loc[num, 'text'] = temp_text


print(train_df)
print(val_df)
print(test_df)
train_df.to_csv(os.path.join(base_path, 'train.csv'), index=False)
val_df.to_csv(os.path.join(base_path, 'val.csv'), index=False)
test_df.to_csv(os.path.join(base_path, 'test.csv'), index=False)