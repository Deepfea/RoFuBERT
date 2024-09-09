import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import jieba
import torch
from torch import nn
from lime.lime_text import LimeTextExplainer
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
b_size = 200
dataset_names = ['cail', 'news', 'weibo']
max_lengths = [512, 64, 256]

class LIMExplainer:
    def __init__(self, model, tokenizer, dataset_name):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        index = dataset_names.index(dataset_name)
        self.max_length = max_lengths[index]
    def predict(self, data):
        # print(data)
        self.model.to(self.device)
        data_token = [self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
                      for text in data]
        # print(len(data_token))
        indexed_tokens = []
        indexed_mask = []
        for input_num in range(len(data_token)):
            temp_mask1 = []
            temp_input = data_token[input_num]['input_ids'].numpy()[0]
            temp_mask = data_token[input_num]['attention_mask'].numpy()[0]
            indexed_tokens.append(temp_input)
            temp_mask1.append(temp_mask)
            indexed_mask.append(temp_mask1)
        indexed_tokens = np.array(indexed_tokens)
        indexed_mask = np.array(indexed_mask)
        # print(indexed_tokens.shape)
        # print(indexed_mask.shape)
        tokens_loader = DataLoader(indexed_tokens, batch_size=b_size)
        mask_loader = DataLoader(indexed_mask, batch_size=b_size)
        for token_idx, tokens_tensor in enumerate(tokens_loader):
            for mask_idx, mask_tensor in enumerate(mask_loader):
                if token_idx != mask_idx:
                    continue
                tokens_tensor = tokens_tensor.to(self.device)
                # print(tokens_tensor.shape)
                mask_tensor = mask_tensor.to(self.device)
                # print(mask_tensor.shape)
                with torch.no_grad():
                    outputs = self.model(input_id=tokens_tensor, mask=mask_tensor)
                    predictions = outputs.detach().cpu().numpy()
                final = [self.softmax(x) for x in predictions]
                if token_idx == 0:
                    final_arr = np.array(final)
                else:
                    final_arr = np.concatenate((final_arr, np.array(final)), axis=0)
                # print(final_arr.shape)
        return final_arr

    def softmax(self, it):
        exps = np.exp(np.array(it))
        return exps / np.sum(exps)

    def split_string(self, string):
        cut_result = jieba.cut(string, cut_all=False)
        cut_result = list(cut_result)
        return cut_result

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.linear = nn.Linear(768, len(labels))
        self.relu = nn.ReLU()

    def forward(self, input_ids, mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask, return_dict=False)
        linear_output = self.linear(pooled_output)
        final_layer = self.relu(linear_output)
        return final_layer

def cal_lime_value(tokenizer, load_path2, model_name, dataset_name, labels, data_df):
    load_model_path = os.path.join(load_path2, model_name,  'best_' + dataset_name + '_' + model_name + '.pt')
    model = torch.load(load_model_path)
    label_names = []
    for label_num in range(len(labels)):
        label_names.append(label_num)
    predictor = LIMExplainer(model, tokenizer, dataset_name)
    explainer = LimeTextExplainer(class_names=labels, split_expression=predictor.split_string)
    data_total_num = len(data_df)
    # print("Example start.")
    results = []
    for data_num in tqdm(range(data_total_num)):

        temp_example = data_df.loc[data_num, 'text']
        temp_label = data_df.loc[data_num, 'label']
        temp_example = str(temp_example)
        print(temp_example)
        temp = predictor.split_string(temp_example)
        exp = explainer.explain_instance(text_instance=temp_example, classifier_fn=predictor.predict, num_features=len(temp),
                                     num_samples=200, labels=label_names)
        result = exp.local_exp[temp_label]
        results.append(result)
    # print("Example done")
    return results




# pretrained_model = "/media/usr/external/home/usr/project/project2_data/model/bert_base_chinese_shap"
# tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
# model_path = '/media/usr/external/home/usr/project/project2_data/step0_train_and_test/bert_base_chinese/best_cail_bert_base_chinese.pt'
# model = torch.load(model_path)
#
# t1 = "赔偿被害人全部经济损失共计人民币10万元，并取得被害人的谅解。"
# t2 = "经审理查明：2013年10月12日，被告人尹某某亲属与被害人达成民事赔偿协议，"
# t3 = '出被告人尹某某系累犯，应当从重处罚；已达成赔偿协议，取得'
# texts = [t1, t2, t3]
# labels = ['盗窃', '危险驾驶', '故意伤害', '交通肇事', '走私、贩卖、运输、制造毒品',
#               '容留他人吸毒', '寻衅滋事', '抢劫', '非法持有毒品', '滥伐林木']
# predictor = LIMExplainer(model, tokenizer)
# explainer = LimeTextExplainer(class_names=labels, split_expression=predictor.split_string)
# to_use = texts[-1:]
# print(to_use)
# for i, example in enumerate(to_use):
#     logging.info(f"Example {i+1}/{len(to_use)} start")
#     temp = predictor.split_string(example)
#     exp = explainer.explain_instance(text_instance=example, classifier_fn=predictor.predict, num_features=len(temp), num_samples=len(temp)*50, labels=labels_name)
#     logging.info(f"Example {i + 1}/{len(to_use)} done")
#     print(exp.local_exp[1])  # 1是label



