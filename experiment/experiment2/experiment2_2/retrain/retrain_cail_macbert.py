from torch.optim import Adam
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch import nn

from experiment.experiment2.experiment2_2.create_dataset import MyDataset

model_name = 'macbert_base_chinese'
dataset_name = 'cail'
load_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
tokenizer = BertTokenizer.from_pretrained(load_path)
model = BertModel.from_pretrained(load_path)
labels = ['盗窃', '危险驾驶', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品',
              '容留他人吸毒', '寻衅滋事', '抢劫', '非法持有毒品', '滥伐林木']

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, len(labels))
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

# def evaluate(model_name, dataset_name, dataset_path, fuzz_path, retrain_path):
#
#     test_data = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
#     test_dataset = MyDataset(test_data, tokenizer, dataset_name)
#     print('测试数据数量：' + str(len(test_data)))
#
#     add_data = pd.read_csv(os.path.join(fuzz_path, model_name, dataset_name, 'add.csv'))
#     print('额外的数据数量：' + str(len(add_data)))
#     add_text_list = add_data['mutant']
#     add_label_list = add_data['label']
#     merge_dt_dict = {'text': add_text_list, 'label': add_label_list}
#     data_df = pd.DataFrame(merge_dt_dict)
#     add_dataset = MyDataset(data_df, tokenizer, dataset_name)
#
#
#     model_load_path = os.path.join(retrain_path, model_name, dataset_name, 'best_' + dataset_name + '_' + model_name + '.pt')
#     model = torch.load(model_load_path)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
#     test_loader = DataLoader(test_dataset, batch_size=128)
#     total_acc_test = 0
#     with torch.no_grad():
#         for test_input, test_label in test_loader:
#             input_id = test_input['input_ids'].squeeze(1).to(device)
#             mask = test_input['attention_mask'].to(device)
#             test_label = test_label.to(device)
#             output = model(input_id, mask)
#             acc = (output.argmax(dim=1) == test_label).sum().item()
#             total_acc_test += acc
#     print(f'Test Accuracy: {total_acc_test / len(test_data): .4f}')
#
#     model.eval()
#     test_loader = DataLoader(add_dataset, batch_size=128)
#     total_acc_test = 0
#     with torch.no_grad():
#         for test_input, test_label in test_loader:
#             input_id = test_input['input_ids'].squeeze(1).to(device)
#             mask = test_input['attention_mask'].to(device)
#             test_label = test_label.to(device)
#             output = model(input_id, mask)
#             acc = (output.argmax(dim=1) == test_label).sum().item()
#             total_acc_test += acc
#     print(f'Add Accuracy: {total_acc_test / len(data_df): .4f}')

def retrain(model_name, dataset_name, save_path):
    coverage_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    val_data_path = os.path.join(save_path, model_name, dataset_name, 'val.csv')
    val_data = pd.read_csv(val_data_path)
    val_dataset = MyDataset(val_data, tokenizer, dataset_name)
    for coverage_rate_num in range(len(coverage_rate_list)):
        temp_coverage_rate = coverage_rate_list[coverage_rate_num]
        save_rate_path = os.path.join(save_path, model_name, dataset_name, 'retrain_model', str(temp_coverage_rate))
        train_data_path = os.path.join(save_path, model_name, dataset_name, 'coverage_data', str(temp_coverage_rate) + '_retrain.csv')
        train_data = pd.read_csv(train_data_path)
        print('训练数据数量：' + str(len(train_data)))
        print('验证数据数量：' + str(len(val_data)))
        train_dataset = MyDataset(train_data, tokenizer, dataset_name)
        epoch = 5
        lr = 1e-5
        batch_size = 32

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        random_seed = 20240121
        setup_seed(random_seed)

        # 定义模型
        model = BertClassifier()

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)
        model = model.to(device)
        criterion = criterion.to(device)

        # 构建数据集
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 训练
        best_dev_acc = 0
        for epoch_num in range(epoch):
            total_acc_train = 0
            total_loss_train = 0
            for inputs, labels in tqdm(train_loader):
                input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
                masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
                labels = labels.to(device)
                output = model(input_ids, masks)
                batch_loss = criterion(output, labels)
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_train += acc
                total_loss_train += batch_loss.item()

            # ----------- 验证模型 -----------
            model.eval()
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():
                for inputs, labels in dev_loader:
                    input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
                    masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
                    labels = labels.to(device)
                    output = model(input_ids, masks)

                    batch_loss = criterion(output, labels)
                    acc = (output.argmax(dim=1) == labels).sum().item()
                    total_acc_val += acc
                    total_loss_val += batch_loss.item()

                print(f'''Epochs: {epoch_num + 1}
                  | Train Loss: {total_loss_train / len(train_dataset): .3f}
                  | Train Accuracy: {total_acc_train / len(train_dataset): .3f}
                  | Val Loss: {total_loss_val / len(val_dataset): .3f}
                  | Val Accuracy: {total_acc_val / len(val_dataset): .3f}''')

                # 保存最优的模型
                if total_acc_val / len(val_dataset) > best_dev_acc:
                    best_dev_acc = total_acc_val / len(val_dataset)
                    if not os.path.exists(save_rate_path):
                        os.makedirs(save_rate_path)
                    save_model_path = os.path.join(save_rate_path, 'best_' + dataset_name + '_' + model_name + '.pt')
                    torch.save(model, save_model_path)

            model.train()

        # 保存最后的模型
        if not os.path.exists(save_rate_path):
            os.makedirs(save_rate_path)
        save_model_path = os.path.join(save_rate_path, 'final_' + dataset_name + '_' + model_name + '.pt')
        torch.save(model, save_model_path)

if __name__ == '__main__':
    pass