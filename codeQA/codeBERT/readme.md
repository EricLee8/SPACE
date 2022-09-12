# Training CodeBERT on CodeQA Dataset

## Prepare Data
First, you should download the dataset from [Google Drive](https://drive.google.com/drive/folders/1i04sJNUHwMuDfMV2UfWeQG-Uv8MRw_qh?usp=sharing). Unzip it and move it to `./data`.

After this, you should have `./data/python` and `./data/java`.

## Build tree-sitter
You shold then build tree-sitter to be ready for training. run:
```
cd parser
bash build.sh
cd ..
```
You are supposed to see file `my-languages.so` in folder `./parser` after this command.

## Start Training
We have prepared shell script for training, to train on Python dataset, run:
```
bash python_script_adv.sh 0 spacepython
```
where the first argument is the CUDA_ID, and the second is where to store the output. You can change these two arguments as you like.

To train on Java dataset, run:
```
bash java_script_adv.sh 0 spacejava
```

To train the baseline models, you can run:
```
bash python_script.sh 0 bslpython
bash java_script.sh 0 bsljava
```
