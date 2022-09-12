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

import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
import multiprocessing
cpu_cont = 16

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_poses)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
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
        dfg=[]
    return code_tokens,dfg

# for entity-only adversarial training                                     
def extract_dataflow_entity_adv(code, parser, lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass
    #obtain dataflow 
    ori_code=code
    tree = parser[0].parse(bytes(code,'utf8'))
    root_node = tree.root_node
    tokens_index=tree_to_token_index(root_node)
    code=code.split('\n')
    code_tokens=[index_to_code_token(x,code) for x in tokens_index]
    index_to_code={}
    for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
        index_to_code[index]=(idx,code)
    variable_poses = tree_to_variable_poses(tree, ori_code, code_tokens, lang)
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
    return code_tokens,dfg,variable_poses

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,
                 idx,
                 var_poses,
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx=position_idx
        self.dfg_to_code=dfg_to_code
        self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url
        self.idx=idx
        self.var_poses=var_poses
        
        
def convert_examples_to_features(item):
    js,tokenizer,args=item
    #code
    parser=parsers[args.lang]
    #extract data flow
    if 'function' in js.keys():
        code = js['function']
    else:
        code = js['original_string']
    if args.entity_adv:
        ori_code_tokens,dfg,variable_poses=extract_dataflow_entity_adv(code,parser,args.lang)
    else:
        ori_code_tokens,dfg=extract_dataflow(code,parser,args.lang)
    if args.codebert: dfg=[]
    code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(ori_code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]  
    #truncating
    dfg_len = len(dfg)
    code_tokens=code_tokens[:args.code_length+args.data_flow_length-2-min(dfg_len,args.data_flow_length)]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg=dfg[:args.code_length+args.data_flow_length-len(code_tokens)]
    code_tokens+=[x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    code_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.code_length+args.data_flow_length-len(code_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    code_ids+=[tokenizer.pad_token_id]*padding_length    
    #adversarial variable positions
    if args.entity_adv:
        var_poses=[[variable_poses[idx]] * (len(tokenizer.tokenize('@ '+x)[1:]) if idx!=0\
         else len(tokenizer.tokenize(x))) for idx,x in enumerate(ori_code_tokens)]
        var_poses=[y for x in var_poses for y in x]
        var_poses=var_poses[:args.code_length+args.data_flow_length-2-min(dfg_len,args.data_flow_length)]
        var_poses=[0]+var_poses+[0]
        var_poses+=[0 for _ in dfg]
        var_poses+=[0]*padding_length
        # to make the var index increasing and continuous, since there are big numbers of truncated variables
        unique_var_idx_mapping = {}
        for pos in var_poses:
            if pos not in unique_var_idx_mapping.keys():
                new_var_id = len(unique_var_idx_mapping)
                unique_var_idx_mapping[pos] = new_var_id
        for idx in range(len(var_poses)):
            if var_poses[idx] > 0:
                var_poses[idx] = unique_var_idx_mapping[var_poses[idx]]
    else:
        var_poses=None

    #reindex
    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'],js['url'],var_poses)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        self.training = 'train' in file_path
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'_{}_{}_{}{}'.format(args.lang,\
             args.code_length, args.data_flow_length, '_enadv' if args.entity_adv else '')+'.pkl'
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
        else:
            self.examples = []
            data=[]
            with open(file_path) as f:
                for line in f:
                    line=line.strip()
                    js=json.loads(line)
                    data.append((js,tokenizer,args))
            logger.info("Creating features from {}...".format(file_path))
            # self.examples=[convert_examples_to_features(x) for x in tqdm(data[:5],total=len(data[:5]))]
            self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            pickle.dump(self.examples,open(cache_file,'wb'))
            
        if self.training:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("index: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))
                logger.info("idx: {}".format(example.idx))
                logger.info("var_poses: {}".format(example.var_poses))
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
                            self.args.code_length+self.args.data_flow_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].code_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
        
        rtv = [torch.tensor(self.examples[item].code_ids),
              torch.tensor(attn_mask),
              torch.tensor(self.examples[item].position_idx), 
              torch.tensor(self.examples[item].nl_ids)]
        if self.args.entity_adv and self.training:
            rtv.append(torch.tensor(self.examples[item].var_poses))
        return rtv


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after amp initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    if args.fp16:
        scaler = GradScaler()
    
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)  
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            nl_inputs = batch[3].to(args.device)
            if args.entity_adv:
                var_poses = batch[4].to(args.device)

            if isinstance(model, torch.nn.DataParallel):
                embeds_init = model.module.encoder.embeddings.word_embeddings(batch[0])
            else:
                embeds_init=model.encoder.embeddings.word_embeddings(code_inputs)

            delta = torch.zeros_like(embeds_init) # (bsz, seqlen, hsz)
            if args.entity_adv:
                zeros = torch.zeros_like(var_poses) # more efficient
                bsz, _, hsz = embeds_init.shape
                for bidx in range(bsz):
                    cur_var_poses = var_poses[bidx] # (slen)
                    cur_zeros = zeros[bidx]
                    cur_max_index = torch.max(cur_var_poses).item()
                    v_deltas = torch.zeros(cur_max_index, hsz, device=args.device)
                    for vidx in range(1, cur_max_index + 1):
                        indices = torch.where(cur_var_poses==vidx, cur_var_poses, cur_zeros).nonzero().squeeze(-1) # (cur_var_nums)
                        if indices.shape[0] == 0: # not found
                            continue
                        
                        indices = indices.detach().cpu().tolist()
                        if isinstance(indices, int):
                            indices = [indices]
                        for index in indices:
                            delta[bidx, index, :] = v_deltas[vidx-1]

            # the main loop
            cur_loss = 0.0
            loss_fct = CrossEntropyLoss()
            for astep in range(args.adv_steps):
                # (0) forward
                delta.requires_grad_()

                if args.fp16:
                    with autocast():
                        #get code and nl vectors
                        code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx,delta=delta)
                        nl_vec = model(nl_inputs=nl_inputs)
                        #calculate scores and loss
                        scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
                        loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
                        if args.n_gpu > 1:
                            loss = loss.mean()
                        loss = loss / args.adv_steps
                        tr_loss += loss.item()
                        cur_loss += loss.item()
                        scaler.scale(loss).backward()
                else:
                    #get code and nl vectors
                    code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx,delta=delta)
                    nl_vec = model(nl_inputs=nl_inputs)
                    #calculate scores and loss
                    scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
                    loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
                    if args.n_gpu > 1:
                        loss = loss.mean()
                    loss = loss / args.adv_steps
                    tr_loss += loss.item()
                    cur_loss += loss.item()
                    loss.backward()

                if astep == args.adv_steps - 1:
                    # further updates on delta
                    break

                delta_grad = delta.grad.clone().detach()

                if args.norm_type == "l2":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
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

                if isinstance(model, torch.nn.DataParallel):
                    embeds_init = model.module.encoder.embeddings.word_embeddings(batch[0])
                else:
                    embeds_init = model.encoder.embeddings.word_embeddings(code_inputs)

            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #update parameters
            if args.fp16:
                with autocast():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['mrr']>best_mrr:
            best_mrr=results['mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = '{}_{}_{}_{}{}'.format(args.lang, args.save_name,\
             args.code_length, args.data_flow_length, '_enadv' if args.entity_adv else '')
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
        
        #test
        evaluate(args, model, tokenizer,args.test_data_file, pool)


def evaluate(args, model, tokenizer, file_name, pool, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running {} *****".format("Evaluation" if "valid" in file_name else "Test"))
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[3].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        attn_mask = batch[1].to(args.device)
        position_idx =batch[2].to(args.device)
        with torch.no_grad():
            code_vec= model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx)
            code_vecs.append(code_vec.cpu().numpy())

    model.train()
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    
    result = {
        "mrr":float(np.mean(ranks))
    }

    logger.info("***** {} results *****".format("Eval" if "valid" in file_name else "Test"))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    print("MRR in {} set:".format("Eval" if "valid" in file_name else "Test"), result['mrr'])

    return result
                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lang", default='python', type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--cuda', type=int, default=-1,
                        help='cuda number when using single GPU, -1 for multi-GPU training.')
    parser.add_argument('--save_name', type=str, default='bsl',
                        help="The name of the save folder")
    parser.add_argument('--fp16', type=int, default=1)
    parser.add_argument('--codebert', type=int, default=0, help='whether to use CodeBERT')

    # adversarial training
    parser.add_argument('--adv_lr', type=float, default=1e-4)
    parser.add_argument('--adv_steps', type=int, default=2, help="should be at least 1")
    parser.add_argument('--norm_type', type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument('--adv_max_norm', type=float, default=0, help="set to 0 to be unlimited")
    parser.add_argument('--entity_adv', type=int, default=0,\
         help='whether to perform adversarial attack only on specific entities.')
    
    pool = multiprocessing.Pool(cpu_cont)
    
    #print arguments
    args = parser.parse_args()

    args.output_dir = "{}saved_models".format("" if args.codebert else "graph_")
    args.config_name = "microsoft/{}codebert-base".format("" if args.codebert else "graph")
    args.model_name_or_path = "microsoft/{}codebert-base".format("" if args.codebert else "graph")
    args.tokenizer_name = "microsoft/{}codebert-base".format("" if args.codebert else "graph")
    if args.codebert:
        args.code_length = args.code_length + args.data_flow_length
        args.data_flow_length = 0
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # Setup CUDA, GPU
    device = torch.device("cuda{}".format((":"+str(args.cuda)) if args.cuda != -1 else "")\
         if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if args.cuda == -1 else 1
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    #build model
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)    
    model=Model(model,args)
    logger.info("Training/evaluation parameters %s", args)
    model.to(args.device)
    
    # Training
    if args.do_train:
        train(args, model, tokenizer, pool)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = '{}_{}_{}_{}{}/model.bin'.format(args.lang, args.save_name,\
             args.code_length, args.data_flow_length, '_enadv' if args.entity_adv else '')
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = '{}_{}_{}_{}{}/model.bin'.format(args.lang, args.save_name,\
             args.code_length, args.data_flow_length, '_enadv' if args.entity_adv else '')
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()