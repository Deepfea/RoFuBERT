import numpy as np
import os
import pandas as pd
def del_seed_in_testing(dataset_name, model_name, load_step1_path):
    load_path = os.path.join(load_step1_path, model_name, dataset_name)
    seeds = pd.read_csv(os.path.join(load_path, 'step0_test_seeds.csv'))
    test_data = pd.read_csv(os.path.join(load_path, 'step0_test.csv'))
    test_output = np.load(os.path.join(load_path, 'step0_test_outputs.npy'))
    del_arr = []
    for seed_num in range(len(seeds)):
        seed_data_num = seeds.loc[seed_num, 'data_num']
        del_arr.append(seed_data_num)
    del_arr = np.array(del_arr)
    test_data = test_data.drop(del_arr, axis=0).reset_index(drop=True)
    test_output = np.delete(test_output, del_arr, axis=0)

    next_turn_data_df = pd.read_csv(os.path.join(load_path, 'step2_next_turn_seeds.csv'))
    next_turn_output_arr = np.load(os.path.join(load_path, 'step2_next_turn_seeds_outputs.npy'), allow_pickle=True)
    if len(next_turn_output_arr) != 0:
        test_data = pd.concat([test_data, next_turn_data_df], ignore_index=True)
        test_output = np.concatenate((test_output, next_turn_output_arr), axis=0)

    test_data.to_csv(os.path.join(load_path, 'step0_test.csv'), index=False)
    np.save(os.path.join(load_path, 'step0_test_outputs.npy'), test_output)


if __name__ == '__main__':

    pass







