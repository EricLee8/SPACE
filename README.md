# Codes for COLING 2022
This repository contains the official codes for our paper at COLING 2022: [Semantic-Preserving Adversarial Code Comprehension](https://aclanthology.org/2022.coling-1.267/).

## Overview
We conduct our experiments on three datasets: Defects4J for Defect Detection, CodeSearchNet for Natural Language Code Search and CodeQA for Question Answering over Source Code.

__You can find codes and instructions in each folder corresponds to each dataset:__

[Defect Detection](./defectDetection)

[Natural Language Code Search](./nlCodeSearch)

[Question Answering over Code](./codeQA)

## Dependencies and Environment
To install the dependencies, please run:
```
pip install -r requirements.txt
```
Besides, we conduct our experiments on the following environment:
```
torch: 1.10.2
python: 3.7.9
CUDA Version: 11.4
GPU: RTX 3090 24G
```
We strongly recommand that you run the experiments on the same environment to ensure the reproductivity.

## Citation
If you find our paper and repository useful, please cite us in your paper:
```
@inproceedings{li-etal-2022-semantic,
    title = "Semantic-Preserving Adversarial Code Comprehension",
    author = "Li, Yiyang  and
      Wu, Hongqiu  and
      Zhao, Hai",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.267",
    pages = "3017--3028",
}
```
