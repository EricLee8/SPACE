# Codes and Data for CodeSearchNet Dataset

## Prepare Data
First, you should get and preprocess the data by running:
```
bash preprocess.sh
```
This will automatically download the data and preprocess it. After that, you are supposed to see `codebase.jsonl`, `train.jsonl`, `valid.jsonl` and `test.jsonl` in each `./dataset/lang/` folder for six programming languages (lang $\in$ [go, java, javascript, php, python, ruby]).

## Build tree-sitter
Then you should build the tree-sitter parser by running:
```
cd parser
bash build.sh
cd ..
```
You are supposed to see file `my-languages.so` in folder `./parser` after this command.

## Start Training
To train SPACE on GraphCodeBERT, you can run the following command:
```
bash run_adv.sh go 5e-5 0
bash run_adv.sh python 5e-4 0
bash run_adv.sh php 1e-4 0
bash run_adv.sh ruby 5e-5 0
bash run_adv.sh javascript 5e-5 0
bash run_adv.sh java 5e-5 0
```
To train SPACE on CodeBERT, run:
```
bash run_adv.sh go 5e-5 1
bash run_adv.sh python 5e-5 1
bash run_adv.sh php 5e-5 1
bash run_adv.sh ruby 5e-4 1
bash run_adv.sh javascript 5e-5 1
bash run_adv.sh java 5e-5 1
```
To train the baseline models, run:
```
bash run.sh lang [0|1]
```
here lang $\in$ [go, python, php, ruby, javascript, java]. The second arguments means `0` for GraphCodeBERT and `1` for CodeBERT.