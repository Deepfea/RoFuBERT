from experiment.experiment2.experiment2_2.cal_coverage import cal_coverage, save_coverage_data, get_coverage_data, \
    show_bucket_num, rank_coverage_data
# from experiment.experiment2.experiment2_2.collect_all_retrain_data import get_and_save_retrain_data, collect_all_retrain_score
from experiment.experiment2.experiment2_2.evaluate_model import get_adv_data, evaluate, evaluate_ori
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


def run_get_retrain_data(dataset_name, model_name, labels, train_model_path, fuzz_path, experiment1_path, save_path):
    print(dataset_name)
    print(model_name)
    # get_and_save_retrain_data(dataset_name, model_name, fuzz_path, save_path)
    # collect_all_retrain_score(dataset_name, model_name, labels, train_model_path, fuzz_path, save_path)
    # show_bucket_num(dataset_name, model_name, save_path)
    # cal_coverage(save_path, model_name, dataset_name)
    # save_coverage_data(dataset_name, model_name, save_path)
    # rank_coverage_data(dataset_name, model_name, save_path)
    # get_coverage_data(dataset_name, model_name, save_path)
    # #  手动运行retrain文件下的py文件来重训练模型
    # get_adv_data(model_name, dataset_name, experiment1_path, save_path)
    # evaluate(model_name, dataset_name, save_path)
    evaluate_ori(model_name, dataset_name, save_path)


if __name__ == '__main__':
    pass


