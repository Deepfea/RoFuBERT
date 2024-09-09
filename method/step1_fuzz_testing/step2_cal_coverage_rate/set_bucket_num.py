def set_coverage_bucket(dataset_name):
    if dataset_name == 'weibo':
        bucket_num = 200
    elif dataset_name == 'news':
        bucket_num = 200
    else:
        bucket_num = 200
    return bucket_num
