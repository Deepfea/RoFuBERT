import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd
from torch import nn

from experiment.experiment2.experiment2_2.create_dataset import MyDataset

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

def softmax(it):
    exps = np.exp(np.array(it))
    return exps / np.sum(exps)

def get_data(dataset_name, model_name, load_ori_model_path, save_path):
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    print("加载数据集......")
    adv_data = pd.read_csv(os.path.join(save_path, model_name, dataset_name, 'adv.csv'))
    label_list = adv_data['label']
    print("加载数据集完成")

    ori_model_path = os.path.join(load_ori_model_path, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')
    retrained_model_path = os.path.join(save_path, model_name, dataset_name, 'retrain_model', '0.6', 'best_' + dataset_name + '_' + model_name + '.pt')

    test_dataset = MyDataset(adv_data, tokenizer, dataset_name)

    model = torch.load(ori_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = temp_npy
            else:
                total_npy = np.concatenate((total_npy, temp_npy), axis=0)
    output_list = []
    # for temp_num in range(len(total_npy)):
    #     temp_npy = total_npy[temp_num]
    #     temp_npy = softmax(temp_npy)
    #     output_list.append(temp_npy)
    # output_arr = np.array(output_list)
    output_arr1 = total_npy
    label_arr1 = np.array(label_list)
    # print(output_arr1.shape)
    # np.save(os.path.join(save_path, 'step0_test_outputs.npy'), output_arr)


    model = torch.load(retrained_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            if test_flag == 0:
                test_flag = 1
                total_npy = temp_npy
            else:
                total_npy = np.concatenate((total_npy, temp_npy), axis=0)
    output_list = []
    # for temp_num in range(len(total_npy)):
    #     temp_npy = total_npy[temp_num]
    #     temp_npy = softmax(temp_npy)
    #     output_list.append(temp_npy)
    # output_arr = np.array(output_list)
    output_arr2 = total_npy
    label_arr2 = np.array(label_list)
    # print(output_arr2.shape)


    return output_arr1, label_arr1, output_arr2, label_arr2


# 加载数据
def cal_T_SNE(dataset_name, model_name, load_ori_model_path, save_path):
    type_list = ['Orignal', 'Retrained']
    data1, label1, data2, label2 = get_data(dataset_name, model_name, load_ori_model_path, save_path)

    # n_samples = len(data)
    # n_features = len(data[0])
    ts = TSNE(n_components=2, init='pca', random_state=0)
    if dataset_name == 'weibo':
        name1 = 'Weibo'
    elif dataset_name == 'news':
        name1 = 'THUCNews'
    else:
        name1 = 'CAIL2018'
    if model_name == 'bert_base_chinese':
        name2 = 'BERT'
    elif model_name == 'roberta_base_chinese':
        name2 = 'RoBERTa'
    else:
        name2 = 'MacBERT'

    tittle = name1 + '_' + name2

    for num in range(len(type_list)):
        print(type_list[num])
        if type_list[num] == 'Orignal':
            data = data1
            label = label1
        else:
            data = data2
            label = label2
        result = ts.fit_transform(data)
        # 调用函数，绘制图像
        fig = plot_embedding(result, label, tittle + '_' + type_list[num] + '_Model')
        fig.savefig(os.path.join(save_path, tittle + '_' + type_list[num] + '_Model' + '.png'), dpi=400)
        # 显示图像
        plt.show()

def get_shape(labels):
    shape_list = ['■', '▲', '♦', '▰', '♥', '⬣', '●', '♣', '♠', '♪']
    label_list = []
    for i in range(len(labels)):
        label_list.append(shape_list[labels[i]])
    return label_list



# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    labels = get_shape(label)
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签△
        if data[i, 0] < 0 or data[i, 0] > 1 or data[i, 1] < 0 or data[i, 1] > 1:
            continue
        if label[i] == 1:
            temp_color = plt.cm.Set3(10)
        else:
            temp_color = plt.cm.Set3(label[i])

        # plt.text(data[i, 0], data[i, 1], labels[i], bbox=dict(facecolor=temp_color, alpha=0.5),
        #          fontdict={'size': 10})
        plt.text(data[i, 0], data[i, 1], labels[i], color=temp_color,
                 fontdict={'size': 10})

        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
        #          fontdict={'weight':  'bold',  'size': 10})

    # ticks_list = [-0.05, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.05]
    # xticks = plt.xticks()
    # plt.xticks(xticks[0][1:], xticks[1])
    #
    plt.xticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1])
    plt.yticks([-0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1])
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.get_xaxis().set_visible(False)
    # plt.yticks([-0.1, 0.0,0.2,0.4,0.6,0.8,1.0, 1.1])
    plt.title(title, fontsize=14)
    # 返回值
    return fig

# 主函数
if __name__ == '__main__':
    pass
