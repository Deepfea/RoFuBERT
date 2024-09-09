from torch.utils.data import Dataset
dataset_names = ['cail', 'news', 'weibo']
max_lengths = [256, 64, 256]

class MyDataset(Dataset):
    def __init__(self, df, tokenizer, dataset_name):
        index = dataset_names.index(dataset_name)
        # print(max_lengths[index])
        # print(max_lengths[index])
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                max_length=max_lengths[index],  # 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class mutantDataset(Dataset):
    def __init__(self, df, tokenizer, dataset_name):
        index = dataset_names.index(dataset_name)
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                max_length=max_lengths[index],  # 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt")
                      for text in df['mutant']]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)