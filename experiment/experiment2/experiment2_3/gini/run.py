from experiment.experiment2.experiment2_3.evaluate_model import get_adv_data, evaluate
from experiment.experiment2.experiment2_3.gini.cal_deepgini_value import cal_deepGini
from experiment.experiment2.experiment2_3.gini.select_test_data import select_test_data
from torch import nn

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

def run_gini(dataset_name, model_name, dataset_path, load_step_path, load_experiment2_1_path, save_step_path):
    print(dataset_name)
    print(model_name)
    # cal_deepGini(dataset_name, model_name, dataset_path, load_step_path, save_step_path)
    # select_test_data(labels, dataset_name, model_name, save_step_path)
    # get_adv_data(model_name, dataset_name, load_experiment2_1_path, save_step_path)
    # #手动训练文件
    evaluate(model_name, dataset_name, save_step_path)

if __name__ == '__main__':
    pass