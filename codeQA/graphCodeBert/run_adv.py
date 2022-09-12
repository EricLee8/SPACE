# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
from collections import OrderedDict, Counter
import torch.nn as nn
from model import Seq2Seq
from graphCodeBert import GraphCodeBert
from tqdm import tqdm, trange
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, cached_path, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from eval.bleu import corpus_bleu
from eval.rouge import Rouge
from eval.meteor import Meteor

from parser import DFG_java, DFG_python
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_mask)
from tree_sitter import Language, Parser
dfg_function={
    'java': DFG_java,
    'python': DFG_python
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source_question,
                 source_code,
                 target,
                 ):
        self.idx = idx
        self.source_question = source_question
        self.source_code = source_code
        self.target = target

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser, lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow 
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        print("Error occured...")
        dfg=[]
    return code_tokens,dfg


def read_examples(filename, stage):
    """Read examples from filename."""
    # filename: e.g. $data_dir/$lang/train/
    # stage: e.g. train
    examples=[]
    idx = 0
    codefile = os.path.join(filename, stage + ".code.original")
    quesfile = os.path.join(filename, stage + ".question")
    ansfile = os.path.join(filename, stage + ".answer")
    with open(codefile,encoding="utf-8") as code_f:
        with open(ansfile, encoding="utf-8") as ans_f:
            with open(quesfile, encoding="utf-8") as ques_f:
                for codeline, quesline, ansline in zip(code_f,ques_f,ans_f):
                    code = codeline.strip()
                    question = quesline.strip()
                    ans = ansline.strip()
                    examples.append(
                        Example(
                                idx = idx,
                                # source= question + " " + code,
                                source_question = question,
                                source_code = code,
                                target = ans,
                                )
                    )
                    idx += 1
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 position_idx,
                 source_mask,
                 dfg_to_code,
                 dfg_to_dfg,
                 target_ids,
                 target_mask,
                 adv_masks=None
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.source_mask = source_mask
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.target_ids = target_ids
        self.target_mask = target_mask
        self.adv_masks = adv_masks  


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists("caches"):
        os.mkdir("caches")
    cache_dir = os.path.join("caches", '{}_{}{}_{}.features'.format(args.lang, stage,\
         '_entity' if args.entity_adv and stage=='train' else '', args.max_source_length))
    if os.path.exists(cache_dir):
        features = torch.load(cache_dir)
        return features

    parser = parsers[args.lang]

    features = []
    for example_index, example in tqdm(enumerate(examples), total=len(examples)):
        #source
        source_question_tokens = tokenizer.tokenize(example.source_question)[:args.max_question_length]
        
        #extract data flow
        ori_code_tokens, dfg = extract_dataflow(example.source_code, parser, args.lang)
        dfg_len, question_len = len(dfg), len(source_question_tokens)
        code_tokens = [tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx, x in enumerate(ori_code_tokens)]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens[i]))    
        code_tokens = [y for x in code_tokens for y in x]
        #truncating and padding
        code_tokens = code_tokens[:args.max_source_length + args.data_flow_length - 3\
             - min(dfg_len,args.data_flow_length) - question_len][:args.max_source_length - 3 - question_len]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token] + source_question_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg = dfg[:args.max_source_length + args.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for _ in dfg]
        source_ids += [tokenizer.unk_token_id for _ in dfg]
        source_mask += [0 for _ in dfg]
        padding_length = args.max_source_length + args.data_flow_length - len(source_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        #reindex
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]

        #adversarial masks
        if args.entity_adv:
            adv_masks = [0] * len(source_ids)
            entity_mapping = {}
            for se in dfg_to_code:
                start, end = se
                if end >= args.max_source_length - 1:
                    break
                cur_entity = tokenizer.convert_tokens_to_string(source_tokens[start: end])
                if cur_entity not in entity_mapping.keys():
                    entity_mapping[cur_entity] = len(entity_mapping) + 1
                for i in range(start, end):
                    adv_masks[i] = entity_mapping[cur_entity]
        else:
            adv_masks=None

        #target
        if stage=="test" or stage=='dev_examples':
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length  
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("position_idx: {}".format(' '.join(map(str, position_idx))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_ids))))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
                logger.info("adv_masks: {}".format(adv_masks))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 position_idx,
                 source_mask,
                 dfg_to_code,
                 dfg_to_dfg, 
                 target_ids,
                 target_mask,
                 adv_masks,
            )
        )
    torch.save(features, cache_dir)
    return features


class TextDataset(Dataset):
    def __init__(self, features, tokenizer, args, training=True, out_adv_mask=False):
        self.features = features
        self.tokenizer = tokenizer
        self.args = args
        self.training = training
        self.out_adv_mask = out_adv_mask
                
    def __len__(self):
        return len(self.features)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask = np.zeros((self.args.max_source_length + self.args.data_flow_length,
                            self.args.max_source_length + self.args.data_flow_length), dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.features[item].position_idx])
        max_length=sum([i!=1 for i in self.features[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        #special tokens attend to all tokens
        for idx, i in enumerate(self.features[item].source_ids):
            if i in [0,2]:
                attn_mask[idx, :max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.features[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index, a:b] = True
                attn_mask[a:b, idx+node_index] = True
        #nodes attend to adjacent nodes 
        for idx, nodes in enumerate(self.features[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index < len(self.features[item].position_idx):
                    attn_mask[idx+node_index, a+node_index] = True  
        rtv = [torch.tensor(self.features[item].source_ids),
                torch.tensor(self.features[item].position_idx),
                torch.tensor(self.features[item].source_mask),
                torch.tensor(attn_mask)]
        if self.training:
            rtv += [torch.tensor(self.features[item].target_ids),
                torch.tensor(self.features[item].target_mask)]
        if self.args.entity_adv and self.out_adv_mask:
            rtv.append(torch.tensor(self.features[item].adv_masks))
        return rtv


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--cuda', type=int, default=-1,
                        help="cuda id for running the code, -1 for multi-gpu training")
    parser.add_argument('--fp16', type=int, default=1,
                        help="whether to use amp for accelerating")
    parser.add_argument('--lang', type=str, default='java')
    parser.add_argument('--grid', type=int, default=0)

    # graph arguments
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--max_question_length", default=128, type=int)

    # adversarial training
    parser.add_argument('--adv_lr', type=float, default=1e-4)
    parser.add_argument('--adv_steps', type=int, default=2, help="should be at least 1")
    parser.add_argument('--norm_type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0, help="set to 0 to be unlimited")
    parser.add_argument('--entity_adv', type=int, default=0,\
         help='whether to perform adversarial attack only on specific entities.')
    parser.add_argument('--entity_lr', type=float, default=0.0,\
         help="adversarial learning rate for entities.")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    log_file = os.path.join(args.output_dir, 'log.txt')
    if args.load_model_path:
        logfile = logging.FileHandler(log_file, 'a')
    else:
        logfile = logging.FileHandler(log_file, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        if args.cuda == -1:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)

        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    #budild model
    inner_encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    encoder = GraphCodeBert(inner_encoder, config, tokenizer, args)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        pass # TODO
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename, 'train')
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        train_data = TextDataset(train_features, tokenizer, args, training=True, out_adv_mask=True)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        if args.fp16:
            scaler = GradScaler()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0, 0, 0, 0, 1e6 
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                if args.entity_adv:
                    source_ids,position_idx,source_mask,attn_mask,target_ids,target_mask,adv_masks = batch
                else:
                    source_ids,position_idx,source_mask,attn_mask,target_ids,target_mask = batch

                embeds_init=model.encoder.encoder.embeddings.word_embeddings(source_ids) # (bsz, slen, hsz)
                delta = torch.zeros_like(embeds_init) # (bsz, seqlen, hsz)
                if args.entity_adv:
                    adv_masks = (adv_masks > 0).float()
                    adv_masks = adv_masks.unsqueeze(-1).expand_as(delta)

                for astep in range(args.adv_steps):
                    delta.requires_grad_()
                    if args.fp16:
                        with autocast():
                            loss,_,_ = model(source_ids=source_ids,position_idx=position_idx,source_mask=source_mask,\
                                 attn_mask=attn_mask,target_ids=target_ids,target_mask=target_mask,delta=delta)
                            if args.n_gpu > 1:
                                loss = loss.mean() # mean() to average on multi-gpu.
                            if args.gradient_accumulation_steps > 1:
                                loss = loss / args.gradient_accumulation_steps
                            loss = loss / args.adv_steps
                            tr_loss += loss.item()
                            scaler.scale(loss).backward()
                    else:
                        loss,_,_ = model(source_ids=source_ids,position_idx=position_idx,source_mask=source_mask,\
                             attn_mask=attn_mask,target_ids=target_ids,target_mask=target_mask,delta=delta)
                        if args.n_gpu > 1:
                            loss = loss.mean() # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        loss = loss / args.adv_steps
                        tr_loss += loss.item()
                        loss.backward()

                    if astep == args.adv_steps - 1:
                        # further updates on delta
                        break

                    delta_grad = delta.grad.clone().detach()

                    if args.norm_type == "l2":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        if args.entity_lr > 0.0:
                            delta = delta + args.entity_lr * delta_grad * adv_masks / denorm
                            delta = (delta + args.adv_lr * delta_grad * (1-adv_masks) / denorm).detach()
                        else:
                            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                        if args.adv_max_norm > 0:
                            delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                            exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                            reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                        + (1-exceed_mask)).view(-1, 1, 1)
                            delta = (delta * reweights).detach()
                    elif args.norm_type == "linf":
                        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                        denorm = torch.clamp(denorm, min=1e-8)
                        delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                        if args.adv_max_norm > 0:
                            delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                        
                    if args.entity_adv and args.entity_lr == 0:
                        delta = delta * adv_masks

                    embeds_init = model.encoder.encoder.embeddings.word_embeddings(source_ids)

                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    if args.fp16:
                        with autocast():
                            scaler.unscale_(optimizer)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            scheduler.step()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    global_step += 1

            if args.do_eval:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename, 'dev')
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
                    eval_data = TextDataset(eval_features, tokenizer, args, training=True)   
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,position_idx,source_mask,attn_mask,target_ids,target_mask = batch                  

                    with torch.no_grad():
                        if args.fp16:
                            with autocast():
                                _,loss,num = model(source_ids=source_ids,position_idx=position_idx,source_mask=source_mask,\
                                     attn_mask=attn_mask,target_ids=target_ids,target_mask=target_mask)
                        else:
                            _,loss,num = model(source_ids=source_ids,position_idx=position_idx,source_mask=source_mask,\
                                 attn_mask=attn_mask,target_ids=target_ids,target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   

                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last-{}'.format(args.max_source_length))
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl-{}'.format(args.max_source_length))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  


                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename, 'dev')
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev_examples')
                    eval_data = TextDataset(eval_features, tokenizer, args, training=False)
                    dev_dataset['dev_bleu']=eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,position_idx,source_mask,attn_mask = batch          
                    with torch.no_grad():
                        if args.fp16:
                            with autocast():
                                preds = model(source_ids=source_ids,position_idx=position_idx,\
                                     source_mask=source_mask,attn_mask=attn_mask)  
                        else:
                            preds = model(source_ids=source_ids,position_idx=position_idx,\
                                 source_mask=source_mask,attn_mask=attn_mask)
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                id = 0
                hypotheses, references = dict(), dict()
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')
                        hypotheses[id] = [ref]
                        references[id] = [gold.target]
                        id += 1

                EM, bleu4, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
                                                                                  references)
                """
                CodeBert smoothed bleu-4 provided in "CodeBERT: A Pre-Trained Model for Programming and Natural Languages" paper. 
                It will produce a much higher bleu score than "A Transformer-based Approach for Source Code Summarization" paper.
                In CodeQA, we use the latter to compute bleu. 
                """
                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                
                if args.grid:
                    info = "dev set: bleu = %.2f | rouge_l = %.2f | meteor = %.2f | EM = %.2f |\
                        Precision = %.2f | Recall = %.2f | F1 = %.2f | " %(bleu4, rouge_l,\
                            meteor, EM, precision, recall, f1)
                    print(info)
                logger.info('dev set: '
                            'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                            (bleu4, rouge_l, meteor) +
                            'EM = %.2f | Precision = %.2f | Recall = %.2f | F1 = %.2f | ' %
                            (EM, precision, recall, f1))
                logger.info("  " + "*" * 20)

                if bleu4>best_bleu:
                    logger.info("  Best bleu:%s",bleu4)
                    logger.info("  "+"*"*20)
                    best_bleu=bleu4
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu-{}'.format(args.max_source_length))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
    
    if args.do_test:
        files=[]
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file, 'test')
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')    
            eval_data = TextDataset(eval_features, tokenizer, args, training=False)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids,position_idx,source_mask,attn_mask = batch                  
                with torch.no_grad():
                    if args.fp16:
                        with autocast():
                            preds = model(source_ids=source_ids,position_idx=position_idx,\
                                 source_mask=source_mask, attn_mask=attn_mask)    
                    else:
                        preds = model(source_ids=source_ids,position_idx=position_idx,\
                             source_mask=source_mask,attn_mask=attn_mask) 
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
                # break #DEBUG
            model.train()
            predictions=[]
            id = 0
            hypotheses, references = dict(), dict()
            with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\n') # ref is actually the hypothesis
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')
                    hypotheses[id] = [ref]
                    references[id] = [gold.target]
                    id += 1

            EM, bleu4, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
                                                                               references)

            # CodeBert smoothed bleu-4
            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx)))
            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
            
            if args.grid:
                info = "test set: bleu = %.2f | rouge_l = %.2f | meteor = %.2f | EM = %.2f |\
                    Precision = %.2f | Recall = %.2f | F1 = %.2f | " %(bleu4, rouge_l,\
                        meteor, EM, precision, recall, f1)
                print(info)
            logger.info('test set: '
                        'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                        (bleu4, rouge_l, meteor) +
                        'EM = %.2f | Precision = %.2f | Recall = %.2f | F1 = %.2f | ' %
                        (EM, precision, recall, f1))
            logger.info("  "+"*"*20)


        # best model test
        best_model_path = os.path.join(args.output_dir, "checkpoint-best-bleu-{}/pytorch_model.bin".format(args.max_source_length))
        logger.info("reload model from {}".format(best_model_path))
        model.load_state_dict(torch.load(best_model_path))
        files=[]
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx,file in enumerate(files):   
            logger.info("Best Model Test file: {}".format(file))
            eval_examples = read_examples(file, 'test')
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            eval_data = TextDataset(eval_features, tokenizer, args, training=False)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval() 
            p=[]
            for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids,position_idx,source_mask,attn_mask= batch                  
                with torch.no_grad():
                    if args.fp16:
                        with autocast():
                            preds = model(source_ids=source_ids,position_idx=position_idx,\
                                 source_mask=source_mask,attn_mask=attn_mask)    
                    else:
                        preds = model(source_ids=source_ids,position_idx=position_idx,\
                             source_mask=source_mask,attn_mask=attn_mask) 
                    for pred in preds:
                        t=pred[0].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                        text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                        p.append(text)
                # break #DEBUG
            model.train()
            predictions=[]
            id = 0
            hypotheses, references = dict(), dict()
            with open(os.path.join(args.output_dir,"test_best_model{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                for ref,gold in zip(p,eval_examples):
                    predictions.append(str(gold.idx)+'\t'+ref)
                    f.write(str(gold.idx)+'\t'+ref+'\n') # ref is actually the hypothesis
                    f1.write(str(gold.idx)+'\t'+gold.target+'\n')
                    hypotheses[id] = [ref]
                    references[id] = [gold.target]
                    id += 1

            EM, bleu4, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
                                                                               references)

            # CodeBert smoothed bleu-4
            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx)))
            dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
            
            if args.grid:
                info = "best model test set: bleu = %.2f | rouge_l = %.2f | meteor = %.2f | EM = %.2f |\
                    Precision = %.2f | Recall = %.2f | F1 = %.2f | " %(bleu4, rouge_l,\
                        meteor, EM, precision, recall, f1)
                print(info)
            logger.info('best model test set: '
                        'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                        (bleu4, rouge_l, meteor) +
                        'EM = %.2f | Precision = %.2f | Recall = %.2f | F1 = %.2f | ' %
                        (EM, precision, recall, f1))
            logger.info("  "+"*"*20)


def eval_accuracies(hypotheses, references):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)


    EM = AverageMeter()
    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    for key in references.keys():
        _EM, _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
                                              references[key])
        EM.update(_EM)
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)

    return EM.avg * 100, bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100

def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    EM, precision, recall, f1 = 0, 0, 0, 0
    for gt in ground_truths:
        _EM, _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            EM, precision, recall, f1 = _EM, _prec, _rec, _f1
    return EM, precision, recall, f1

def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            EM, precision, recall, f1 = 1, 1, 1, 1
    else:
        EM = (normalize_answer(prediction) == normalize_answer(ground_truth))
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return EM, precision, recall, f1

def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))
                
                
if __name__ == "__main__":
    main()


