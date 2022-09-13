# Codes for COLING 2022
This repository contains the official codes for our paper at COLING 2022: [Semantic-Preserving Adversarial Code Comprehension](https://arxiv.org/abs/2108.10015).

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
