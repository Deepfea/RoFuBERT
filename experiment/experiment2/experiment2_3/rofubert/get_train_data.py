import os
import pandas as pd

def get_train_data(dataset_name, model_name, dataset_path, save_step_path):
    save_path = os.path.join(save_step_path, model_name, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_data = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    train_data.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    val_data = pd.read_csv(os.path.join(dataset_path, 'val.csv'))
    val_data.to_csv(os.path.join(save_path, 'val.csv'), index=False)

if __name__ == '__main__':
    pass
