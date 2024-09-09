import pandas as pd
import os

def cal_max_length(data_df):
    max_length = 0
    for i in range(len(data_df)):
        temp_text = data_df.loc[i, 'text']
        if len(temp_text) > max_length:
            max_length = len(temp_text)
    return max_length

def cal_average_length(base_path):
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(base_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
    num = 0
    total_len = 0
    for i in range(len(train_df)):
        temp_text = train_df.loc[i, 'text']
        num += 1
        total_len += len(temp_text)
    for i in range(len(val_df)):
        temp_text = val_df.loc[i, 'text']
        num += 1
        total_len += len(temp_text)
    for i in range(len(test_df)):
        temp_text = test_df.loc[i, 'text']
        num += 1
        total_len += len(temp_text)
    ave_len = total_len / float(num)
    return ave_len

base_path = '/media/usr/external/home/usr/project/project2_data/dataset/CAIL2018_ALL_DATA'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
length = cal_max_length(train_df)
print(length)
val_df = pd.read_csv(os.path.join(base_path, 'val.csv'))
length = cal_max_length(val_df)
print(length)
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
length = cal_max_length(val_df)
print(length)
ave_len = cal_average_length(base_path)
print(ave_len)

base_path = '/media/usr/external/home/usr/project/project2_data/dataset/THUCNews'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
length = cal_max_length(train_df)
print(length)
val_df = pd.read_csv(os.path.join(base_path, 'val.csv'))
length = cal_max_length(val_df)
print(length)
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
length = cal_max_length(val_df)
print(length)
ave_len = cal_average_length(base_path)
print(ave_len)

base_path = '/media/usr/external/home/usr/project/project2_data/dataset/weibo'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
length = cal_max_length(train_df)
print(length)
val_df = pd.read_csv(os.path.join(base_path, 'val.csv'))
length = cal_max_length(val_df)
print(length)
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
length = cal_max_length(val_df)
print(length)
ave_len = cal_average_length(base_path)
print(ave_len)