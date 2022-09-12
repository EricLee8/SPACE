import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, inputs_ids,position_idx,attn_mask,labels=None,delta=None):
        #embedding
        if self.args.codebert: # pure CodeBERT
            inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
            attn_mask=inputs_ids.ne(1)
            position_idx,token_type_ids=None,None
        else:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
            token_type_ids=position_idx.eq(-1).long()
        if delta is not None:
            inputs_embeddings += delta

        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,\
            position_ids=position_idx,token_type_ids=token_type_ids)[0]
        logits=self.classifier(outputs)
        # shape: [batch_size, num_classes]
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
