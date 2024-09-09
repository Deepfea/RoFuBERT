from chinesebert import ChineseBertConfig
from chinesebert import ChineseBertTokenizerFast, ChineseBertModel
from sklearn.metrics import precision_score, auc, roc_curve, f1_score, recall_score
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_cosine_schedule_with_warmup
import warnings
import random
import numpy as np
# import tensorflow as tf
import torch
import time
warnings.filterwarnings('ignore')
import os
seed = 1
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)
torch.manual_seed(seed)


model_name = 'chinesebert_base'
dataset_name = 'cail'
load_path = '/media/usr/external/home/usr/project/project2_data/model/chinesebert_base'
all_labels = ['盗窃', '危险驾驶', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品', '容留他人吸毒', '寻衅滋事', '抢劫', '非法持有毒品', '滥伐林木']
step_path = '/media/usr/external/home/usr/project/project2_data/epxeriment/experiment1/experiment1_1'

dataset_path1 = '/media/usr/external/home/usr/project/project2_data/epxeriment/experiment1/experiment1_1/chinesebert_base'

tokenizer = ChineseBertTokenizerFast.from_pretrained(load_path)
config = ChineseBertConfig.from_pretrained(load_path)
chinese_bert = ChineseBertModel.from_pretrained(load_path, config=config)

def get_data(csv_data_name):
    print(csv_data_name)
    input_ids, input_masks, input_types, input_pinyin, labels = [], [], [], [], []
    maxlen = 256
    dataset_path = dataset_path1

    data = pd.read_csv(os.path.join(dataset_path, csv_data_name))
    title_list = data['text']
    y_list = data['label']
    for num in tqdm(range(len(title_list))):
        title = title_list[num]
        y = y_list[num]
        encode_dict = tokenizer.encode_plus(text=title, max_length=maxlen, padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_types.append(encode_dict['token_type_ids'])
        input_masks.append(encode_dict['attention_mask'])
        input_pinyin.append(encode_dict['pinyin_ids'])
        labels.append(int(y))

    input_ids, input_types, input_masks, input_pinyin, labels = np.array(input_ids), np.array(input_types), np.array(input_masks), np.array(input_pinyin), np.array(labels)
    print(input_ids.shape, input_types.shape, input_masks.shape, input_pinyin.shape, labels.shape,)
    idxes = np.arange(input_ids.shape[0])
    y = labels[idxes[:]]
    return input_ids, input_masks, input_types, y, input_pinyin


model_name_list = ['bert_base_chinese', 'roberta_base_chinese', 'macbert_base_chinese']
adv_name1 = model_name_list[0] + '_' + dataset_name + '_adv.csv'
adv_name2 = model_name_list[1] + '_' + dataset_name + '_adv.csv'
adv_name3 = model_name_list[2] + '_' + dataset_name + '_adv.csv'

input_ids_test1, input_masks_test1, input_types_test1, y_test1, inputs_pinin_test1 = get_data(adv_name1)
input_ids_test2, input_masks_test2, input_types_test2, y_test2, inputs_pinin_test2 = get_data(adv_name2)
input_ids_test3, input_masks_test3, input_types_test3, y_test3, inputs_pinin_test3 = get_data(adv_name3)

test_batch_size = 16

def get_tensor():
    # 测试集（是没有标签的）
    test_data1 = TensorDataset(torch.LongTensor(input_ids_test1),
                              torch.LongTensor(input_masks_test1),
                              torch.LongTensor(input_types_test1),
                              torch.LongTensor(inputs_pinin_test1),
                                torch.LongTensor(y_test1))
    test_sampler = SequentialSampler(test_data1)
    test_loader1 = DataLoader(test_data1, sampler=test_sampler, batch_size=test_batch_size)
    test_data2 = TensorDataset(torch.LongTensor(input_ids_test2),
                              torch.LongTensor(input_masks_test2),
                              torch.LongTensor(input_types_test2),
                              torch.LongTensor(inputs_pinin_test2),
                              torch.LongTensor(y_test2))
    test_sampler = SequentialSampler(test_data2)
    test_loader2 = DataLoader(test_data2, sampler=test_sampler, batch_size=test_batch_size)
    test_data3 = TensorDataset(torch.LongTensor(input_ids_test3),
                              torch.LongTensor(input_masks_test3),
                              torch.LongTensor(input_types_test3),
                              torch.LongTensor(inputs_pinin_test3),
                              torch.LongTensor(y_test3))
    test_sampler = SequentialSampler(test_data3)
    test_loader3 = DataLoader(test_data3, sampler=test_sampler, batch_size=test_batch_size)
    return test_loader1, test_loader2, test_loader3


test_loader1, test_loader2, test_loader3 = get_tensor()

class Bert_Model(nn.Module):
    # 定义model
    # def __init__(self, bert_path, classes=10):
    def __init__(self, classes=len(all_labels)):
        super(Bert_Model, self).__init__()
        self.config = config  # 导入模型超参数
        self.bert = chinese_bert  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pinyin_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pinyin_ids=pinyin_ids)
        global out_pool
        out_pool = outputs['last_hidden_state'][:, 0, :]  # 表示CLS
        logit = self.fc(out_pool)  # [bs, classes]
        return logit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_loader, device, y_test):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, att, tpe, piyi, y) in tqdm(enumerate(data_loader)):
            y_pred = model(input_ids=ids.to(device), attention_mask=att.to(device), token_type_ids=tpe.to(device), pinyin_ids=piyi.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    acc_num = 0
    index_list = []
    for num in range(len(val_pred)):
        if int(val_pred[num]) == int(y_test[num]):
            acc_num += 1
        else:
            index_list.append(num)
    acc = acc_num / float(len(val_pred))
    print(1)
    print(1 - acc)
    return index_list


model_path = os.path.join(step_path, model_name, dataset_name)
model_load_path = os.path.join(model_path, 'best_' + dataset_name + '_' + model_name + '.pt')
model = torch.load(model_load_path)

index_list = evaluate(model, test_loader1, DEVICE, y_test1)

add_data = pd.read_csv(os.path.join(dataset_path1, adv_name1))
avg_modify_rate = 0
for num in range(len(index_list)):
    temp_word = add_data.loc[index_list[num], 'word']
    if temp_word == 'nan':
        continue
    temp_word = str(temp_word)
    temp_word = temp_word.replace("+", "")
    temp_mutant = add_data.loc[index_list[num], 'text']
    temp_rate = float(len(temp_word)) / float(len(temp_mutant))
    avg_modify_rate += temp_rate
avg_modify_rate = avg_modify_rate / float(len(index_list))
print(avg_modify_rate)

index_list = evaluate(model, test_loader2, DEVICE, y_test2)

add_data = pd.read_csv(os.path.join(dataset_path1, adv_name2))
avg_modify_rate = 0
for num in range(len(index_list)):
    temp_word = add_data.loc[index_list[num], 'word']
    if temp_word == 'nan':
        continue
    temp_word = str(temp_word)
    temp_word = temp_word.replace("+", "")
    temp_mutant = add_data.loc[index_list[num], 'text']
    temp_rate = float(len(temp_word)) / float(len(temp_mutant))
    avg_modify_rate += temp_rate
avg_modify_rate = avg_modify_rate / float(len(index_list))
print(avg_modify_rate)

index_list = evaluate(model, test_loader3, DEVICE, y_test3)

add_data = pd.read_csv(os.path.join(dataset_path1, adv_name3))
avg_modify_rate = 0
for num in range(len(index_list)):
    temp_word = add_data.loc[index_list[num], 'word']
    if temp_word == 'nan':
        continue
    temp_word = str(temp_word)
    temp_word = temp_word.replace("+", "")
    temp_mutant = add_data.loc[index_list[num], 'text']
    temp_rate = float(len(temp_word)) / float(len(temp_mutant))
    avg_modify_rate += temp_rate
avg_modify_rate = avg_modify_rate / float(len(index_list))
print(avg_modify_rate)
