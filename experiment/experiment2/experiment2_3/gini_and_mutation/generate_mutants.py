import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from method.step1_fuzz_testing.get_sentence.get_sentence import get_sentence
from method.step1_fuzz_testing.get_word.get_glyph_word import get_glyph_word
from method.step1_fuzz_testing.get_word.get_pinyin_word import get_pinyin_word
from method.step1_fuzz_testing.get_word.get_shuffle_word import get_shuffle_word
from method.step1_fuzz_testing.get_word.get_splitting_word import get_splitting_word
from method.step1_fuzz_testing.get_word.get_synonym_word import get_synonym_word

def get_sentences(fact, temp_segs, synonym_seg_list, shuffle_seg_list, pinyin_seg_list, splitting_seg_list, glyph_seg_list):
    sentences = []
    ori_strs = []
    rep_strs = []
    type_list = []

    x, y, z = get_sentence(fact, temp_segs, synonym_seg_list)
    temp_list = ['synonym'] * len(x)
    type_list += temp_list
    sentences += x
    ori_strs += y
    rep_strs += z

    x, y, z = get_sentence(fact, temp_segs, shuffle_seg_list)
    temp_list = ['shuffle'] * len(x)
    type_list += temp_list
    sentences += x
    ori_strs += y
    rep_strs += z

    x, y, z = get_sentence(fact, temp_segs, pinyin_seg_list)
    temp_list = ['pinyin'] * len(x)
    type_list += temp_list
    sentences += x
    ori_strs += y
    rep_strs += z

    x, y, z = get_sentence(fact, temp_segs, splitting_seg_list)
    temp_list = ['splitting'] * len(x)
    type_list += temp_list
    sentences += x
    ori_strs += y
    rep_strs += z

    x, y, z = get_sentence(fact, temp_segs, glyph_seg_list)
    temp_list = ['glyph'] * len(x)
    type_list += temp_list
    sentences += x
    ori_strs += y
    rep_strs += z

    return sentences, ori_strs, rep_strs, type_list

def generate_mutant1(dataset_name, model_name, load_step_path, importance_num):
    load_seed_path = os.path.join(load_step_path, model_name, dataset_name)
    load_seg_path = os.path.join(load_seed_path, str(importance_num))
    seg_list = np.load(os.path.join(load_seg_path, 'select_segs.npy'), allow_pickle=True)
    seeds = pd.read_csv(os.path.join(load_seed_path, 'final_seed.csv'))
    start_flag = 0
    for seed_num in tqdm(range(len(seeds))):
        temp_segs = list(seg_list[seed_num])
        fact = seeds.loc[seed_num, 'text']
        label = seeds.loc[seed_num, 'label']

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
        sentences, ori_strs, rep_strs, type_list = get_sentences(fact, temp_segs, synonym_seg_list, shuffle_seg_list, pinyin_seg_list, splitting_seg_list, glyph_seg_list)
        # print(len(sentences))
        # print(len(ori_strs))
        # print(len(rep_strs))
        belong_list = []
        fact_list = []
        mutant_list = []
        str_list = []
        word_list = []
        label_list = []
        for sentences_num in range(len(sentences)):
            belong_list.append(seed_num)
            fact_list.append(fact)
            str_list.append(ori_strs[sentences_num])
            word_list.append(rep_strs[sentences_num])
            mutant_list.append(sentences[sentences_num])
            label_list.append(label)
        merge_dt_dict = {'belong': belong_list, 'text': fact_list, 'str': str_list, 'word': word_list,
                         'mutant': mutant_list, 'mutate_tpye': type_list, 'label': label_list}
        data_df = pd.DataFrame(merge_dt_dict)
        if start_flag == 0:
            mutant_df = data_df
            start_flag = 1
        else:
            mutant_df = pd.concat([mutant_df, data_df], ignore_index=True)
    save_path = os.path.join(load_seg_path, 'seed_mutants.csv')
    # if os.path.exists(save_path):
    #     temp_df = pd.read_csv(save_path)
    #     mutant_df = pd.concat([temp_df, mutant_df], ignore_index=True)
    print('当前变异体数量：' + str(len(mutant_df)))
    mutant_df.to_csv(save_path, index=False)

if __name__ == '__main__':

    pass
