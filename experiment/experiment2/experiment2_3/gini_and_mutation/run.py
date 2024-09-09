
from torch import nn

from experiment.experiment2.experiment2_3.evaluate_model import get_adv_data, evaluate
from experiment.experiment2.experiment2_3.gini_and_mutation.cal_test_output import cal_output
from experiment.experiment2.experiment2_3.gini_and_mutation.cal_word_scores import cal_word_scores, select_segs
from experiment.experiment2.experiment2_3.gini_and_mutation.cut_seg import cut_seg
from experiment.experiment2.experiment2_3.gini_and_mutation.generate_mutants import generate_mutant1
from experiment.experiment2.experiment2_3.gini_and_mutation.get_retrain_data import get_retrain_data
from experiment.experiment2.experiment2_3.gini_and_mutation.select_mutants import select_mutants
from experiment.experiment2.experiment2_3.gini_and_mutation.select_seed_data import select_test_data


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

def run_gini_and_mutation(labels, dataset_name, model_name, dataset_path, load_step_path, load_experiment2_1_path,save_step_path):
    print(dataset_name)
    print(model_name)
    if dataset_name == 'weibo':
        important_num = 1
    elif dataset_name == 'news':
        important_num = 2
    else:
        important_num = 12
    # cal_output(dataset_name, model_name, dataset_path, load_step_path, save_step_path)
    # select_test_data(labels, dataset_name, model_name, save_step_path)
    # cut_seg(labels, dataset_name, model_name, save_step_path)
    # cal_word_scores(labels, dataset_name, model_name, load_step_path, save_step_path)
    # select_segs(dataset_name, model_name, save_step_path, important_num)
    # generate_mutant1(dataset_name, model_name, save_step_path, important_num)
    # select_mutants(dataset_name, model_name, load_step_path, save_step_path, important_num)
    # get_retrain_data(labels, dataset_name, model_name, save_step_path, important_num)

    # # 手动重训练
    get_adv_data(model_name, dataset_name, load_experiment2_1_path, save_step_path)
    evaluate(model_name, dataset_name, save_step_path)

if __name__ == '__main__':
    pass
