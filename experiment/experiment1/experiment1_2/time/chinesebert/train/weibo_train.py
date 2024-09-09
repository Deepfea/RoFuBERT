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

time1 = time.time()
model_name = 'chinesebert_base'
dataset_name = 'weibo'
load_path = '/media/usr/external/home/usr/project/project2_data/model/chinesebert_base'
all_labels = ['none', 'like', 'disgust', 'happiness', 'sadness']
dataset_path1 = '/media/usr/external/home/usr/project/project2_data/dataset/weibo'
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

input_ids_train, input_masks_train, input_types_train, y_train, inputs_pinin_train= get_data('train.csv')
input_ids_valid, input_masks_valid, input_types_valid, y_valid, inputs_pinin_valid = get_data('val.csv')
input_ids_test, input_masks_test, input_types_test, y_test, inputs_pinin_test = get_data('test.csv')

train_batch_size = 16  # 太大会出现OOM问题，太小训练数据样本分布与总体差异大，自己衡量
dev_batch_size = 16
test_batch_size = 16

def get_tensor():
    # 训练集
    train_data = TensorDataset(torch.LongTensor(input_ids_train),
                               torch.LongTensor(input_masks_train),
                               torch.LongTensor(input_types_train),
                               torch.LongTensor(inputs_pinin_train),
                               torch.LongTensor(y_train))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    # 验证集
    valid_data = TensorDataset(torch.LongTensor(input_ids_valid),
                              torch.LongTensor(input_masks_valid),
                              torch.LongTensor(input_types_valid),
                               torch.LongTensor(inputs_pinin_valid),
                               torch.LongTensor(y_valid))
    valid_sampler = SequentialSampler(valid_data)
    valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=dev_batch_size)
    # 测试集（是没有标签的）
    test_data = TensorDataset(torch.LongTensor(input_ids_test),
                              torch.LongTensor(input_masks_test),
                              torch.LongTensor(input_types_test),
                              torch.LongTensor(inputs_pinin_test))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)
    return train_loader, valid_loader, test_loader


train_loader, valid_loader, test_loader = get_tensor()

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
EPOCHS = 5
model = Bert_Model().to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=EPOCHS*len(train_loader))

def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, att, tpe, piyi, y) in tqdm(enumerate(data_loader)):
            y_pred = model(input_ids=ids.to(device), attention_mask=att.to(device), token_type_ids=tpe.to(device), pinyin_ids=piyi.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())

    return accuracy_score(val_true, val_pred)  # 返回accuracy

def train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, device, epoch):
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, piyi, y) in enumerate(train_loader):
            y = y.to(device)
            y_pred = model(input_ids=ids.to(device), attention_mask=att.to(device), token_type_ids=tpe.to(device), pinyin_ids=piyi.to(device))
            loss = criterion(y_pred, y)
            optimizer.zero_grad()  # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率变化
            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_loader) // 5) == 0:  # 只打印五次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start))
                print("Learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        model.eval()
        acc = evaluate(model, valid_loader, device)  # 验证模型的性能
        ## 保存最优模型
        if acc > best_acc:
            best_acc = acc
            # torch.save(model.state_dict(), "BestBERTModel.pth")
            step_path = '/media/usr/external/home/usr/project/project2_data/epxeriment/experiment1/experiment1_2/time/chinesebert'
            model_path = os.path.join(step_path, model_name, dataset_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            save_path = os.path.join(model_path, 'best_' + dataset_name + '_' + model_name + '.pt')
            torch.save(model, save_path)

        print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))

train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, DEVICE, EPOCHS)

time2 = time.time()
print(time2)

time3 = time2 - time1
time3 = time3 / float(3600)
print(time3)



