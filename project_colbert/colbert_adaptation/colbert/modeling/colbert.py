import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE
import numpy as np
import pandas as pd
import os
import pickle

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

        # print(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.linear_input1 = nn.Linear(config.hidden_size, dim, bias=False)
        self.linear_input2 = nn.Linear(config.hidden_size, dim, bias=False)

        linear_dim = dim*3
        self.lambda_linear1 = nn.Linear(linear_dim, linear_dim)
        self.lambda_linear2 = nn.Linear(linear_dim, linear_dim)
        self.lambda_linear3 = nn.Linear(linear_dim, 2)
        self.adaptation_linear1 = nn.Linear(self.query_maxlen, self.query_maxlen)
        self.adaptation_linear2 = nn.Linear(self.query_maxlen, self.query_maxlen)
        self.relu = nn.ReLU()

        with open('../data/msmarco_pass/oracle_lambda_train.pkl', 'rb') as f:
            self.lambda_oracle = pickle.load(f)

        with open('../data/msmarco_pass/oracle_lambda_dev.pkl', 'rb') as f:
            self.lambda_oracle_test = pickle.load(f)
        
        # self.lambda_oracle = {qid:[query_tok, query_tok],
        #                       qid:[query_tok, query_tok]
        #                      }
        self.threshold = nn.Threshold(0.9, 0)
        self.init_weights()

    def freeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False

    def freeze_encoder_adaptation(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = False
        for param in self.lambda_linear1.parameters():
            param.requires_grad = False
        for param in self.lambda_linear2.parameters():
            param.requires_grad = False
        for param in self.lambda_linear3.parameters():
            param.requires_grad = False
        
    def predict_lambda(self, Q):
        # contextualized_vector_q = self.query(*Q)
        # contextualized_vector = [batchsize, query_len, token_dim]
        Q_ = self.relu(self.lambda_linear1(self.query_n_input(*Q)))
        Q_ = self.relu(self.lambda_linear2(Q_))
        pred = self.lambda_linear3(Q_)
        # print(pred.shape)
        # pred = [batchsize, query_len, 2]

        return pred

    def lambda_target(self, qid, Q_tok, train=True):
        batch_size = Q_tok.shape[0]
        query_len = Q_tok.shape[1]
        target = torch.zeros_like(Q_tok).to(DEVICE)
        
        if train:
            # train에선 list of qid가 들어옴.
            lambda_oracle = self.lambda_oracle
            
            for batch_idx, query in enumerate(Q_tok):
                try:
                    lambda_ = lambda_oracle[qid[batch_idx]]
                    for query_idx, query_tok in enumerate(query):
                        if query_tok.item() in lambda_:
                            target[batch_idx][query_idx] = 1
                except:
                    pass
        
        else:
            # test에선 하나의 qid만 들어옴. (top1000 reranking 기준)
            lambda_oracle = self.lambda_oracle_test
            for batch_idx, query in enumerate(Q_tok):
                try:
                    lambda_ = lambda_oracle[qid]
                    for query_idx, query_tok in enumerate(query):
                        if query_tok.item() in lambda_:
                            target[batch_idx][query_idx] = 1
                except:
                    # print("no match")
                    pass
        
        # classifier debugging / 전부 1로 예측하는지 확인.
        # target = torch.ones_like(Q_tok).to(DEVICE)

        return target

    def forward(self, qid, Q, D):
        return self.score(qid, Q, D, self.query(*Q), self.doc(*D)) # coil + colbert
       

    def query_n_input(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q) # 768 -> 128
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        
        # Q.shape = [batch, query_len, vector_dim]

        qlayer_11 = bert_output[2][11] # 6-th layer bert input / 768
        qlayer_11 = self.linear_input1(qlayer_11) # 768 -> 128
        qlayer_11 = torch.nn.functional.normalize(qlayer_11, p=2, dim=2)

        qlayer_0 = bert_output[2][0] # 0-th layer bert input / 768
        qlayer_0 = self.linear_input2(qlayer_0) # 768 -> 128
        qlayer_0 = torch.nn.functional.normalize(qlayer_0, p=2, dim=2)
        
        # Q와 qlayer_0를 concat.
        Q_ = torch.cat((Q, qlayer_11, qlayer_0), 2)

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
        # shape -> [batchsize*2] = [64]

    # def lambda_mask(self, qid, Q_V, D_V):
    #     Q_V, D_V = Q_V.to(DEVICE), D_V.to(DEVICE)
    #     batch_size = Q_V.shape[0]
    #     query_len = Q_V.shape[1]
    #     doc_len = D_V.shape[1]

    #     mask = torch.zeros(batch_size, query_len, doc_len).to(DEVICE)
    #     one_mask = torch.ones(1, doc_len).to(DEVICE)
    #     zero_mask = torch.zeros(1, doc_len).to(DEVICE)
    #     # one_mask = 0.8 * torch.ones(1, doc_len).to(DEVICE)
    #     # zero_mask = 0.2 * torch.ones(1, doc_len).to(DEVICE)


    #     for batch_idx, query in enumerate(Q_V):
    #         try:
    #             lambda_ = self.lambda_oracle[qid[batch_idx]]
    #         except:
    #             # will work like ColBERT
    #             lambda_ = []
                
    #         for query_idx, query_tok in enumerate(query):
    #             if query_tok.item() in lambda_:
    #                 mask[batch_idx][query_idx] = one_mask
    #             else:
    #                 mask[batch_idx][query_idx] = zero_mask
    #     return mask.int().float()

    def lambda_mask_adaptation_thirdstage(self, Q, D):
        pred = self.predict_lambda(Q)
        pred = torch.log_softmax(pred, dim=2)

        doc_len = D[0].shape[1]
        batch_size, query_len = pred.shape[0], pred.shape[1]
        # pred.shape = [batch_size, query_len, 2]
        
        ##
        ## class 1의 confidence
        pred = torch.exp(pred) # for actual probabilities
        pred = pred[:, :, 1] # take class 1 confidence
        # shape: [batch_size, query_len]

        ####################
        pred = self.relu(self.adaptation_linear1(pred))
        pred = self.adaptation_linear2(pred)
        ################
        # print(pred.shape)
        # print(pred[0])

        pred = pred * Q[1].to(DEVICE) # [MASK] 부분 prediction 무시.

        pred.unsqueeze_(-1)
        # shape: [batch_size, query_len, 1]
        ##

        ##
        ## 둘중 높은 confidence를 가지는 class.
        # _, pred = torch.max(pred, dim=2)
        # pred.unsqueeze_(-1)
        ##
        

        # 암튼 pred.shape = [batch_size, query_len, 1]이 되도록 바꾸고, mask로 쓸수 있도록 형태 바꿔주면 끝.
        lambda_mask = pred.expand(batch_size, query_len, doc_len)
        return lambda_mask

    def lambda_mask_classifier(self, Q, D): # adaptation 따로 안했을 때 
        pred = self.predict_lambda(Q)
        pred = torch.log_softmax(pred, dim=2)

        doc_len = D[0].shape[1]
        batch_size, query_len = pred.shape[0], pred.shape[1]
        # pred.shape = [batch_size, query_len, 2]
        
        ##
        ## class 1의 confidence
        pred = torch.exp(pred) # for actual probabilities
        pred = pred[:, :, 1] # take class 1 confidence
        # shape: [batch_size, query_len]
        # print(pred[0])
        
        pred = self.threshold(pred)
        
        
        # print(pred[0])
        # print(pred.shape)
        # print(torch.sum(torch.nonzero(pred[0])))
        pred = pred * Q[1].to(DEVICE) # [MASK] 부분 prediction 무시.
        # print(torch.sum(torch.nonzero(pred[0])))

        pred.unsqueeze_(-1)
        # shape: [batch_size, query_len, 1]
        ##

        ##
        ## 둘중 높은 confidence를 가지는 class.
        # _, pred = torch.max(pred, dim=2)
        # pred.unsqueeze_(-1)
        ##

        # 암튼 pred.shape = [batch_size, query_len, 1]이 되도록 바꾸고, mask로 쓸수 있도록 형태 바꿔주면 끝.
        lambda_mask = pred.expand(batch_size, query_len, doc_len)
        return lambda_mask

    def score(self, qid, Q, D, Q_128, D_128):
        Q_V = Q[0]
        D_V = D[0]

        coil_mask = self.coil_mask(Q_V, D_V) # shape: [batchsize, query_len, doc_len]
        token_wise_score = (Q_128 @ D_128.permute(0, 2, 1)) # cosine similarity
        lambda_mask = self.lambda_mask_classifier(Q, D) # shape: [batchsize, query_len, doc_len]

        # Q_V = [batch_size, query_len]
        # Q_128 = [batchsize, query_len, token_dim]



        coil_masked_score = token_wise_score * coil_mask * lambda_mask
        coil_score = coil_masked_score.max(2).values.sum(1)
        
        colbert_masked_score = token_wise_score * (torch.ones_like(lambda_mask) - lambda_mask)
        colbert_score = colbert_masked_score.max(2).values.sum(1)

        return coil_score + colbert_score

    # def forward(self, Q, D):
    #     return self.score(self.query(*Q), self.doc(*D))

    # def score(self, Q, D):
    #     if self.similarity_metric == 'cosine':
    #         return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)


    #     assert self.similarity_metric == 'l2'
    #     return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
