import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def draw_violinplot():
    # make data:
    df = pd.DataFrame(data={
        'type': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
        'value': [0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.9],
        'hue': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    })

    sns.violinplot(x=df['type'], y=df['value'], hue=df['hue'])

    # # plot:
    # fig, ax = plt.subplots()
    #
    # vp = ax.violinplot(D, [2, 4, 6], widths=1,
    #                    showmeans=True, showmedians=True, showextrema=True)
    # # styling:
    # for body in vp['bodies']:
    #     body.set_alpha(0.9)
    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #        ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()


def draw_violin(dataset_name, model_name_list, rofubert_path, argot_path, save_path):
    if dataset_name == 'weibo':
        temp_name = 'Weibo'
    elif dataset_name == 'news':
        temp_name = 'THUCNews'
    else:
        temp_name = 'CAIL2018'
    print(dataset_name)
    df = pd.DataFrame(columns=(temp_name, 'Similarity', 'Method'))  # 生成空的pandas表
    for model_name in model_name_list:
        rofubert_values = np.load(os.path.join(rofubert_path, model_name, dataset_name, 'sim_scores.npy'))
        argot_values = np.load(os.path.join(argot_path, model_name, dataset_name, 'sim_scores.npy'))
        if model_name == 'bert_base_chinese':
            model_name1 = 'BERT'
        elif model_name == 'roberta_base_chinese':
            model_name1 = 'RoBERTa'
        else:
            model_name1 = 'MacBERT'
        for value in tqdm(rofubert_values):
            df = df.append({temp_name: model_name1, 'Similarity': value, 'Method': 'RoFuBERT'}, ignore_index=True)
        for value in tqdm(argot_values):
            df = df.append({temp_name: model_name1, 'Similarity': value, 'Method': 'Argot'}, ignore_index=True)
    total = len(rofubert_values) + len(argot_values)

    plt.figure(figsize=(20, 10))
    sns.set(font_scale=2)
    fig = sns.violinplot(x=df[temp_name], y=df['Similarity'], hue=df['Method'])
    violinplot_fig = fig.get_figure()
    violinplot_fig.savefig(os.path.join(save_path, dataset_name + 'violinplot.png'), dpi=400)
    plt.show()

if __name__ == '__main__':
    pass

