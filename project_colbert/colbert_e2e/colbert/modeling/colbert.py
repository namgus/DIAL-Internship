import string
import torch
import torch.nn as nn
import pickle

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE
import numpy as np
import pandas as pd
import os

class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        config.output_hidden_states = True
        config.return_dict = True

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.linear_input = nn.Linear(config.hidden_size, dim, bias=False)
######
        linear_input = dim*4
        self.lambda_linear1 = nn.Linear(linear_input, linear_input)
        self.lambda_linear2 = nn.Linear(linear_input, linear_input)
        self.lambda_linear3 = nn.Linear(linear_input, 1)
######
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, Q, D):
        
        # return self.score(Q, D, self.query(*Q), self.doc(*D)) # coil + colbert
        return self.score(Q, D, self.query(*Q), self.doc(*D)) # coil & colbert
       
    ###########
    def max_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    ###########
    
    def query_n_input(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q) # 768 -> 128
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        
        # Q.shape = [batch, query_len, vector_dim]

        # qlayer_11 = bert_output[2][11] # 6-th layer bert input / 768
        # qlayer_11 = self.linear_input(qlayer_11) # 768 -> 128
        # qlayer_11 = torch.nn.functional.normalize(qlayer_11, p=2, dim=2)

        # qlayer_0 = bert_output[2][0] # 0-th layer bert input / 768
        # qlayer_0 = self.linear_input(qlayer_0) # 768 -> 128
        # qlayer_0 = torch.nn.functional.normalize(qlayer_0, p=2, dim=2)


        qlayer_7 = bert_output[2][7] # 6-th layer bert input / 768
        qlayer_7 = self.linear_input(qlayer_7) # 768 -> 128
        qlayer_7 = torch.nn.functional.normalize(qlayer_7, p=2, dim=2)
        qlayer_9 = bert_output[2][9] # 6-th layer bert input / 768
        qlayer_9 = self.linear_input(qlayer_9) # 768 -> 128
        qlayer_9 = torch.nn.functional.normalize(qlayer_9, p=2, dim=2)
        qlayer_12 = bert_output[2][12] # 6-th layer bert input / 768
        qlayer_12 = self.linear_input(qlayer_12) # 768 -> 128
        qlayer_12 = torch.nn.functional.normalize(qlayer_12, p=2, dim=2)

        
        # Q와 qlayer_0를 concat.
        Q_ = torch.cat((Q, qlayer_7, qlayer_9, qlayer_12), 2)

        return Q_

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask
        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D
    
    def coil_tok_score(self, Q_V, D_V, Q_128, D_128):
        coil_mask = self.coil_mask(Q_V, D_V).int().float() # shape: [query_len, doc_len]
        token_wise_score = (Q_128 @ D_128.permute(0, 2, 1)) # cosine similarity
        token_wise_score = token_wise_score * coil_mask

        return token_wise_score.max(2).values.sum(1)

    def coil_mask(self, Q_V, D_V):
        Q_V, D_V = Q_V.to(DEVICE), D_V.to(DEVICE)
        batch_size = Q_V.shape[0]
        query_len = Q_V.shape[1]
        doc_len = D_V.shape[1]

        mask = torch.zeros(batch_size, query_len, doc_len).to(DEVICE)
        one_mask = torch.ones(1, doc_len).to(DEVICE)
        zero_mask = torch.zeros(1, doc_len).to(DEVICE)

        for batch_idx, query in enumerate(Q_V):
            for query_idx, query_tok in enumerate(query):
                mask[batch_idx][query_idx] = torch.where(D_V[batch_idx] == query_tok, one_mask, zero_mask)
        return mask.int().float()

    def colbert_score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
            # Q.shape -> [batch, query_len, dim per token]
            # D.shape -> [batch, doc_len, dim per token]
            # Q @ D.permute(0, 2, 1) 
            # == [batch, query_len, dim per token] @ [batch, dim per token, doc_len]
            # == [batch, query len, doc_len]
            # .max(2).values == [batch, query_len]
            # .sum(1) == [batch]
        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def lambda_mask_e2e(self, Q, D):
        pred = self.predict_lambda(Q)
        doc_len = D[0].shape[1]
        batch_size, query_len = pred.shape[0], pred.shape[1]
        # pred.shape = [batch_size, query_len, 1]
        # print(pred[0])
        lambda_mask = pred.expand(batch_size, query_len, doc_len)
        # print(lambda_mask)
        return lambda_mask

    def predict_lambda(self, Q):
        # contextualized_vector_q = self.query(*Q)
        # contextualized_vector = [batchsize, query_len, token_dim]
        Q_ = self.relu(self.lambda_linear1(self.query_n_input(*Q)))
        Q_ = self.relu(self.lambda_linear2(Q_))
        pred = self.sigmoid(self.lambda_linear3(Q_)) # [batchsize, query_len, 1]

        return pred

    def score(self, Q, D, Q_128, D_128):
        Q_V = Q[0]
        D_V = D[0]

        # lambda_predict = self.predict_lambda(Q) # [batchsize, query_len, 1]
        coil_mask = self.coil_mask(Q_V, D_V) # shape: [batchsize, query_len, doc_len]
        # print(Q[0].size())
        token_wise_score = (Q_128 @ D_128.permute(0, 2, 1)) # cosine similarity
        lambda_mask = self.lambda_mask_e2e(Q, D) # shape: [batchsize, query_len, doc_len]

        coil_masked_score = token_wise_score * coil_mask * lambda_mask
        coil_score = coil_masked_score.max(2).values.sum(1)
        colbert_masked_score = token_wise_score * (torch.ones_like(lambda_mask) - lambda_mask)
        colbert_score = colbert_masked_score.max(2).values.sum(1)

        return coil_score + colbert_score

    def score_coil_colbert(self, Q, D, Q_128, D_128):
        Q_V = Q[0]
        D_V = D[0]

        # lambda_predict = self.predict_lambda(Q) # [batchsize, query_len, 1]
        coil_mask = self.coil_mask(Q_V, D_V) # shape: [batchsize, query_len, doc_len]
        token_wise_score = (Q_128 @ D_128.permute(0, 2, 1)) # cosine similarity
        # lambda_mask = self.lambda_mask_e2e(Q, D) # shape: [batchsize, query_len, doc_len]

        # coil_masked_score = token_wise_score * coil_mask
        # coil_score = coil_masked_score.max(2).values.sum(1)

        colcoil_mask = torch.where((coil_mask.sum(-1)==0).unsqueeze(2).expand(coil_mask.shape),torch.ones(coil_mask.shape).to(DEVICE), coil_mask)
        colcoil_masked_score = token_wise_score * colcoil_mask
        colcoil_score = colcoil_masked_score.max(2).values.sum(1)

        return colcoil_score

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
