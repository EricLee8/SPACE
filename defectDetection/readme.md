# Codes and Data for Defects4J Dataset

## Prepare Data
First, you should preprecess the metadata to get the full dataset. To do this, please run:
```
cd dataset
python preprocess.py
cd ..
```
After this, you are supposed to see `train.json`, `valid.json` and `test.json` in folder `./dataset`.

## Build tree-sitter
Then you should build the tree-sitter parser by running:
```
cd parser
bash build.sh
cd ..
```
You are supposed to see file `my-languages.so` in folder `./parser` after this command.

## Start Training
We provide shell script to train our models. You can run to train SPACE:
```
bash run_adv.sh
```
in `run_adv.sh`, you can set `codebert="1"` to train on CodeBERT and `codebert="0"` to train on GraphCodeBERT.

To train the baselines, run:
```
bash run.sh
```
in `run.sh`, you can set `codebert="1"` to train on CodeBERT and `codebert="0"` to train on GraphCodeBERT.
