# RoFuBERT: Guiding Robust Fine-tuning using Fuzz Testing for Chinese BERT-based PLMs

### Overview:

We propose RoFuBERT, a robust fine-tuning framework using fuzz testing for Chinese BERT-based PLMs.
Given a pre-trained model, we first select the seeds in the dataset of downstream NLP tasks. The adversarial samples generated on these seeds are more likely to deceive models and lead to wrong outputs. 
Secondly, we make the mutation strategy for seeds. The mutants generated with seeds can simulate adversarial attacks and discover the vulnerability of the target model. 
Thirdly, we measure the mutant coverage with Local Interpretable Model-Agnostic Explanations (LIME). It evaluates the testing completeness under attacks. 
Finally, we retrain the model as the result of robust improvement.


### Setup:
We conduct the experiments on Pytorch (v1.7.1). 
The physical host is a machine running on the Ubuntu 18.04 system, equipped with one Nvidia RTX 3090 GPU, the Intel i9-10900K(3.7GHz) CPU, and 64GB of RAM.

We utilize Anaconda 3 to manage all of the Python packages. To facilitate reproducibility 
of the Python environment, we release an Anaconda YAML specification file of the libraries 
utilized in the experiments. This allows the user to create a new virtual Python environment 
with all of the packages required to run the code by importing the YAML file. 

### Dataset:
We conduct the experiments on three datasets——Weibo,  THUCNews and CAIL2018 dataset subset.

**Weibo:** It consists of individual user posts from the microblogging platform. 
We collect 5 categories of emotions for experiments, including likeness, disgust, happiness, sadness, and none. 
The data amount in each category is balanced, and the average data length is 28 Chinese words.

**THUCNews:** It , a rich and diverse dataset, offers a wide range of topics for exploration. 
We select 10 categories for our experiments, including home furnishings, stocks, entertainment, gaming, society, technology, politics, sports, education, and finance. 
The data amount in each category is balanced, and the average data length is 20 Chinese words.

**CAIL2018:** It , designed for legal text comprehension and reasoning competitions, is highly relevant to real-world applications. 
We select 10 categories for experiments, including theft, dangerous driving, intentional injury, traffic accidents, drug trafficking, allowing others to use drugs, illegal possession of drugs, provoking trouble, robbery, and indiscriminate logging. 
The data amount in each category is balanced, and the average data length is 178 Chinese words.

### Models:
We use three BERT-based PLMs in Chinese for each dataset, including BERT, RoBERTa, and MacBERT. 
BERT is the first BERT-Based model for the Chinese language. 
RoBERTa improves BERT with the hyperparameters, training data size, and mask mechanism during pre-training.
MacBERT improves on the Whole Word Masking strategy. 
We fine-tune three Chinese tasks on these models, respectively. We set the hidden size to 768 and train each model with the Adam optimization, whose learning rate is 5E-5. The epoch is five. 
We select the model with the highest validation accuracy for the result of fine-tuning.

### Running the code:

Environment Setup: 
````
    1. Setup a Linux environment (not tested for Windows) with an Nvidia GPU containing at least 12GB of memory (less may work, but not tested).   
    2. Download data from the link: https://pan.baidu.com/s/1_814GSOIjcnrADiEfClhoQ?pwd=2024 
    3. Download the open-sourced code, dataset and models.
    4. Create a virtual Python environment using the provided YAML configuration file on Github.
    5. Activate the new virtual Python environment
````
Parameters:

* Dataset(s): the user can select between a few datasets. 

* Model(s): the user can select between a few models. 

**Running RoFuBERT:**
In order to run RoFuBERT, run the file under approach. Parameter options refer to the paper.
