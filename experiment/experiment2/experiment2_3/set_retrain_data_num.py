

def set_retrain_data_num(dataset_name):
    print(dataset_name)
    if dataset_name == 'cail' or dataset_name == 'news':
        data_num = 4000
    else:
        data_num = 1000
    return data_num