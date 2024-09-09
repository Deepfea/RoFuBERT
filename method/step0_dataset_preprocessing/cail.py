import json
import os
import pandas as pd
from flatten_json import flatten
from collections import Counter

# start = 0
from tqdm import tqdm

base_path = '/media/usr/external/home/usr/project/project2_data/dataset/CAIL2018_ALL_DATA'
# name_list = os.listdir(os.path.join(base_path, 'ini_data'))
# for file_name in name_list:
#     print(file_name)
#     file_path = os.path.join(base_path, 'ini_data', file_name)
#     file = open(file_path, 'r', encoding='UTF-8')
#     complex_json_data = []
#     for line in file.readlines():
#         dic = json.loads(line)
#         complex_json_data.append(dic)
#     flat_json_data = [flatten(item) for item in complex_json_data]
#     df_complex = pd.DataFrame(flat_json_data)
#     select_col = ['fact', 'meta_accusation_0']
#     flat_data = df_complex[select_col]
#     if start == 0:
#         dataset_df = flat_data
#         start = 1
#     else:
#         print(len(dataset_df))
#         print(len(flat_data))
#         dataset_df = pd.concat([dataset_df, flat_data], ignore_index=True)
#         print(len(dataset_df))
#     print(dataset_df)
#
# values_list = df_complex['meta_accusation_0'].tolist()
# counts = Counter(values_list)
# print(counts)
#
# # 统计整个数据集已有的不同类别的数据量，方便做成我们要的数据集
#
labels = ['盗窃', '危险驾驶', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品',
         '容留他人吸毒', '寻衅滋事', '抢劫', '非法持有毒品', '滥伐林木']
# for label_i in range(len(labels)):
#     label = []
#     label.append(labels[label_i])
#     filtered_df = dataset_df[dataset_df['meta_accusation_0'].isin(label)]
#     filtered_df = filtered_df.reset_index(drop=True)
#     values_list = filtered_df['meta_accusation_0'].tolist()
#     # 使用Counter统计各元素的数量
#     counts = Counter(values_list)
#     print(counts)
#     temp_list = []
#     for num in range(len(filtered_df)):
#         temp_list.append(label_i)
#     filtered_df['label'] = temp_list
#     filtered_df.to_csv(os.path.join(base_path, str(label_i) + '.csv'), index=False)

# 生成并保存训练集、验证集和测试集

def select(data_df, max):
    text_list = []
    label_list = []
    meta_accusation_0_list = []
    for data_num in tqdm(range(len(data_df))):
        temp_text = data_df.loc[data_num, 'fact']
        temp_label = data_df.loc[data_num, 'label']
        temp_meta_accusation_0 = data_df.loc[data_num, 'meta_accusation_0']
        if len(temp_text) > max:
            continue
        text_list.append(temp_text)
        label_list.append(temp_label)
        meta_accusation_0_list.append(temp_meta_accusation_0)
    merge_dt_dict = {'text': text_list, 'label': label_list, 'meta_accusation_0': meta_accusation_0_list}
    data_df = pd.DataFrame(merge_dt_dict)
    return data_df

# max_list = [90, 100, 110, 150, 110, 110, 200, 220, 170, 220]
max_list = [250, 250, 250, 250, 250, 250, 250, 250, 250, 250]

for label_i in range(len(labels)):
    temp_df = pd.read_csv(os.path.join(base_path, str(label_i) + '.csv'))
    temp_df = temp_df.sample(frac=1).reset_index(drop=True)
    temp_df = select(temp_df, max_list[label_i])
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
print(train_df)
print(val_df)
print(test_df)

print("处理训练集的r和n：")
for num in range(len(train_df)):
    s = train_df.loc[num, 'text']
    s = s.replace('\r', '')
    s = s.replace('\n', '')
    s = s.replace('\t', '')
    s = s.replace(' ', '')
    train_df.loc[num, 'text'] = s

print("处理验证集的r和n：")
for num in range(len(val_df)):
    s = val_df.loc[num, 'text']
    s = s.replace('\r', '')
    s = s.replace('\n', '')
    s = s.replace('\t', '')
    s = s.replace(' ', '')
    val_df.loc[num, 'text'] = s

print("处理测试集的r和n：")
for num in range(len(test_df)):
    s = test_df.loc[num, 'text']
    s = s.replace('\r', '')
    s = s.replace('\n', '')
    s = s.replace('\t', '')
    s = s.replace(' ', '')
    test_df.loc[num, 'text'] = s

train_df.to_csv(os.path.join(base_path, 'train.csv'), index=False)
val_df.to_csv(os.path.join(base_path, 'val.csv'), index=False)
test_df.to_csv(os.path.join(base_path, 'test.csv'), index=False)



