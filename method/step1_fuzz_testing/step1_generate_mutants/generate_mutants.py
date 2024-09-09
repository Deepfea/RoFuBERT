import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
from torch import nn

from method.step1_fuzz_testing.create_dataset import mutantDataset
from method.step1_fuzz_testing.get_sentence.get_sentence import get_sentence
from method.step1_fuzz_testing.get_word.get_glyph_word import get_glyph_word
from method.step1_fuzz_testing.get_word.get_pinyin_word import get_pinyin_word
from method.step1_fuzz_testing.get_word.get_shuffle_word import get_shuffle_word
from method.step1_fuzz_testing.get_word.get_splitting_word import get_splitting_word
from method.step1_fuzz_testing.get_word.get_synonym_word import get_synonym_word
from method.step1_fuzz_testing.mlm_model import get_n_1_word, get_n_2_word, generate_sentence

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

def cal_output(dataset_name, model_name, load_path, data):
    model_load_path = os.path.join(load_path, model_name, 'best_' + dataset_name + '_' + model_name + '.pt')
    model = torch.load(model_load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    load_token_path = os.path.join('/media/usr/external/home/usr/project/project2_data/model', model_name)
    tokenizer = BertTokenizer.from_pretrained(load_token_path)
    dataset = mutantDataset(data, tokenizer, dataset_name)
    # print(len(dataset))
    test_loader = DataLoader(dataset, batch_size=128)
    test_flag = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            temp_len = len(test_label)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            output = model(input_id, mask)
            temp_npy = torch.squeeze(output).cpu().detach().numpy()
            # print(temp_npy.shape)
            if temp_len == 1:
                temp_list = []
                temp_list.append(temp_npy)
                temp_npy = np.array(temp_list)
            # print(temp_npy.shape)
            if test_flag == 0:
                test_flag = 1
                total_npy = temp_npy
            else:
                total_npy = np.concatenate((total_npy, temp_npy), axis=0)
    output_list = []
    for temp_num in range(len(total_npy)):
        temp_npy = total_npy[temp_num]
        temp_npy = softmax(temp_npy)
        output_list.append(temp_npy)
    output_arr = np.array(output_list)
    return output_arr

def generate_mutant(labels, dataset_name, model_name, load_model_path, load_step_path):
    load_seed_path = os.path.join(load_step_path, model_name, dataset_name)
    seg_list = np.load(os.path.join(load_seed_path, 'step1_test_seeds_segs.npy'), allow_pickle=True)
    seeds = pd.read_csv(os.path.join(load_seed_path, 'step0_test_seeds.csv'))
    seeds_output = np.load(os.path.join(load_seed_path, 'step0_test_seeds_outputs.npy'), allow_pickle=True)
    start_flag = 0
    mutant_output = []
    for seed_num in range(len(seeds)):
        belong_list = []
        fact_list = []
        mutant_list = []
        str_list = []
        word_list = []
        label_list = []
        fact = seeds.loc[seed_num, 'text']
        label = seeds.loc[seed_num, 'label']
        print(fact)
        print(label)
        temp_segs = seg_list[seed_num]
        for seg_num in tqdm(range(len(temp_segs))):
            temp_seg = temp_segs[seg_num]
            # print(temp_seg)
            word1_list = get_n_1_word(fact, temp_seg, n1=1)
            word2_list = get_n_2_word(fact, temp_seg, n1=1, n2=1)
            temp_1_sentences = generate_sentence(fact, temp_seg, word1_list)
            temp_2_sentences = generate_sentence(fact, temp_seg, word2_list)
            for temp_1_sentences_num in range(len(temp_1_sentences)):
                belong_list.append(seed_num)
                fact_list.append(fact)
                str_list.append(temp_seg)
                word_list.append(word1_list[temp_1_sentences_num])
                mutant_list.append(temp_1_sentences[temp_1_sentences_num])
                label_list.append(label)
            for temp_2_sentences_num in range(len(temp_2_sentences)):
                belong_list.append(seed_num)
                fact_list.append(fact)
                str_list.append(temp_seg)
                word_list.append(word2_list[temp_2_sentences_num])
                mutant_list.append(temp_2_sentences[temp_2_sentences_num])
                label_list.append(label)
        merge_dt_dict = {'belong': belong_list, 'text': fact_list, 'str': str_list,
                         'word': word_list, 'mutant': mutant_list, 'label': label_list}
        data_df = pd.DataFrame(merge_dt_dict)
        if len(data_df) == 0:
            continue
        output = cal_output(dataset_name, model_name, load_model_path, data_df)
        seed_output = seeds_output[seed_num]
        temp_index1 = np.argmax(seed_output)
        get_num = 0
        temp_list = []
        for output_num in range(len(output)):
            temp_output = output[output_num]
            temp_index2 = np.argmax(temp_output)
            if temp_index1 == temp_index2:
                temp_list.append(output_num)
                get_num += 1
            else:
                mutant_output.append(temp_output)
        data_df = data_df.drop(temp_list).reset_index(drop=True)
        if start_flag == 0:
            mutant_df = data_df
            start_flag = 1
        else:
            mutant_df = pd.concat([mutant_df, data_df], ignore_index=True)
    print(mutant_df)
    mutant_output = np.array(mutant_output)
    print(len(mutant_output))
    mutant_df.to_csv(os.path.join(load_seed_path, 'step1_seed_mutants.csv'), index=False)
    np.save(os.path.join(load_seed_path, 'step1_seed_mutants_outputs.npy'), mutant_output)

def get_sentences(fact, temp_segs, synonym_seg_list, shuffle_seg_list, pinyin_seg_list, splitting_seg_list, glyph_seg_list):
    sentences = []
    ori_strs = []
    rep_strs = []
    x, y, z = get_sentence(fact, temp_segs, synonym_seg_list)
    sentences += x
    ori_strs += y
    rep_strs += z
    x, y, z = get_sentence(fact, temp_segs, shuffle_seg_list)
    sentences += x
    ori_strs += y
    rep_strs += z
    x, y, z = get_sentence(fact, temp_segs, pinyin_seg_list)
    sentences += x
    ori_strs += y
    rep_strs += z
    x, y, z = get_sentence(fact, temp_segs, splitting_seg_list)
    sentences += x
    ori_strs += y
    rep_strs += z
    x, y, z = get_sentence(fact, temp_segs, glyph_seg_list)
    sentences += x
    ori_strs += y
    rep_strs += z
    return sentences, ori_strs, rep_strs

def generate_mutant1(labels, dataset_name, model_name, load_model_path, load_step_path, counter):
    load_seed_path = os.path.join(load_step_path, model_name, dataset_name)
    seg_list = np.load(os.path.join(load_seed_path, 'step1_test_seeds_segs.npy'), allow_pickle=True)
    seeds = pd.read_csv(os.path.join(load_seed_path, 'step0_test_seeds.csv'))
    seeds_output = np.load(os.path.join(load_seed_path, 'step0_test_seeds_outputs.npy'), allow_pickle=True)
    start_flag = 0
    mutant_output = []
    for seed_num in range(len(seeds)):
        temp_segs = list(seg_list[seed_num])
        fact = seeds.loc[seed_num, 'text']
        label = seeds.loc[seed_num, 'label']
        seed_counter = seeds.loc[seed_num, 'counter']
        ori_num = seeds.loc[seed_num, 'origin']
        # print(fact)
        # print(len(temp_segs))
        belong_list = []
        fact_list = []
        mutant_list = []
        str_list = []
        word_list = []
        label_list = []
        synonym_seg_list = []
        shuffle_seg_list = []
        pinyin_seg_list = []
        splitting_seg_list = []
        glyph_seg_list = []
        for seg_num in range(len(temp_segs)):
            temp_seg = temp_segs[seg_num]
            # print(temp_seg)
            synonym_seg = get_synonym_word(fact, temp_seg)
            synonym_seg_list.append(synonym_seg)
            # print(synonym_seg_list)

            shuffle_seg = get_shuffle_word(temp_seg)
            shuffle_seg_list.append(shuffle_seg)
            # print(shuffle_seg)

            pinyin_seg = get_pinyin_word(temp_seg)
            pinyin_seg_list.append(pinyin_seg)
            # print(pinyin_seg)

            splitting_seg = get_splitting_word(temp_seg)
            splitting_seg_list.append(splitting_seg)
            # print(splitting_seg)

            glyph_seg = get_glyph_word(temp_seg)
            glyph_seg_list.append(glyph_seg)
            # print(glyph_seg)
        # print(fact)
        sentences, ori_strs, rep_strs = get_sentences(fact, temp_segs, synonym_seg_list, shuffle_seg_list, pinyin_seg_list, splitting_seg_list, glyph_seg_list)
        # print(sentences)
        # print(len(sentences))
        # print(len(ori_strs))
        # print(len(rep_strs))
        counter_list = []
        ori_num_list = []
        if seed_counter == 0:
            temp_counter = 0
        else:
            temp_counter = counter
        for sentences_num in range(len(sentences)):
            belong_list.append(seed_num)
            fact_list.append(fact)
            str_list.append(ori_strs[sentences_num])
            word_list.append(rep_strs[sentences_num])
            mutant_list.append(sentences[sentences_num])
            label_list.append(label)
            counter_list.append(temp_counter)
            ori_num_list.append(ori_num)
        merge_dt_dict = {'belong': belong_list, 'text': fact_list, 'str': str_list,
                         'word': word_list, 'mutant': mutant_list, 'label': label_list, 'counter': counter_list,
                         'origin': ori_num_list}
        data_df = pd.DataFrame(merge_dt_dict)
        # print(data_df)
        if len(data_df) == 0:
            continue
        output = cal_output(dataset_name, model_name, load_model_path, data_df)
        seed_output = seeds_output[seed_num]
        temp_index1 = np.argmax(seed_output)
        get_num = 0
        temp_list = []
        for output_num in range(len(output)):
            temp_output = output[output_num]
            temp_index2 = np.argmax(temp_output)
            if temp_index1 == temp_index2:
                temp_list.append(output_num)
                get_num += 1
            else:
                mutant_output.append(temp_output)
        data_df = data_df.drop(temp_list).reset_index(drop=True)
        if start_flag == 0:
            mutant_df = data_df
            start_flag = 1
        else:
            mutant_df = pd.concat([mutant_df, data_df], ignore_index=True)
    # print(mutant_df)
    mutant_output = np.array(mutant_output)
    # print(len(mutant_output))

    mutant_df.to_csv(os.path.join(load_seed_path, 'step1_seed_mutants.csv'), index=False)
    np.save(os.path.join(load_seed_path, 'step1_seed_mutants_outputs.npy'), mutant_output)

if __name__ == '__main__':

    pass
