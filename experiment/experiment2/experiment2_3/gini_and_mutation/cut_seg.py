import numpy as np
import os
import pandas as pd
import jieba

def cut_seg(labels, dataset_name, model_name,load_step_path):
    load_value_path = os.path.join(load_step_path, model_name, dataset_name)
    test_data = pd.read_csv(os.path.join(load_value_path, 'final_seed.csv'))
    seg_list = []
    for data_num in range(len(test_data)):
        # print(data_num)
        temp_string = test_data.loc[data_num, 'text']
        cut_result = jieba.cut(temp_string, cut_all=False)
        cut_result = list(cut_result)
        seg_list.append(cut_result)
    seg_arr = np.array(seg_list)
    # print(output_arr)
    np.save(os.path.join(load_value_path, 'seed_segs.npy'), seg_arr)


if __name__ == '__main__':

    pass