import os
import random
import time
import torch
import torch.nn as nn

from itertools import accumulate
from math import ceil

from colbert.utils.runs import Run
from colbert.utils.utils import print_message

from colbert.evaluation.metrics import Metrics
from colbert.evaluation.ranking_logger import RankingLogger
from colbert.modeling.inference import ModelInference

from colbert.evaluation.slow import slow_rerank
from colbert.parameters import DEVICE

import pickle


def evaluate(args):
    args.inference = ModelInference(args.colbert, amp=args.amp)
    qrels, queries, topK_pids = args.qrels, args.queries, args.topK_pids

    depth = args.depth
    collection = args.collection
    if collection is None:
        topK_docs = args.topK_docs

    def qid2passages(qid):
        if collection is not None:
            return [collection[pid] for pid in topK_pids[qid][:depth]]
        else:
            return topK_docs[qid][:depth]

    metrics = Metrics(mrr_depths={10, 100}, recall_depths={50, 200, 1000},
                      success_depths={5, 10, 20, 50, 100, 1000},
                      total_queries=len(queries))

    ranking_logger = RankingLogger(Run.path, qrels=qrels)

    args.milliseconds = []

    with ranking_logger.context('ranking.tsv', also_save_annotations=(qrels is not None)) as rlogger:
        with torch.no_grad():
            keys = sorted(list(queries.keys()))
            random.shuffle(keys)

            for query_idx, qid in enumerate(keys):
                query = queries[qid]

                print_message(query_idx, qid, query, '\n')

                if qrels and args.shortcircuit and len(set.intersection(set(qrels[qid]), set(topK_pids[qid]))) == 0:
                    continue

                ranking = slow_rerank(args, query, topK_pids[qid], qid2passages(qid))

                rlogger.log(qid, ranking, [0, 1])

                if qrels:
                    metrics.add(query_idx, qid, ranking, qrels[qid])

                    for i, (score, pid, passage) in enumerate(ranking):
                        if pid in qrels[qid]:
                            print("\n#> Found", pid, "at position", i+1, "with score", score)
                            print(passage)
                            break

                    metrics.print_metrics(query_idx)
                    metrics.log(query_idx)

                print_message("#> checkpoint['batch'] =", args.checkpoint['batch'], '\n')
                print("rlogger.filename =", rlogger.filename)

                if len(args.milliseconds) > 1:
                    print('Slow-Ranking Avg Latency =', sum(args.milliseconds[1:]) / len(args.milliseconds[1:]))

                print("\n\n")

        print("\n\n")
        # print('Avg Latency =', sum(args.milliseconds[1:]) / len(args.milliseconds[1:]))
        print("\n\n")

    print('\n\n')
    if qrels:
        assert query_idx + 1 == len(keys) == len(set(keys))
        metrics.output_final_metrics(os.path.join(Run.path, 'ranking.metrics'), query_idx, len(queries))
    print('\n\n')


def evaluate_lambda_term(args):
    inference = ModelInference(args.colbert, amp=args.amp)
    queries = args.test_queries
    keys = sorted(list(queries.keys()))
    colbert = args.colbert
    
    token_count = {}    

    pred_1_count = 0
    total = 0

    threshold = nn.Threshold(0.6, 0)


    with torch.no_grad():
        for query_idx, qid in enumerate(keys):
            query = queries[qid]
            print(query)

            Q = inference.queryFromText([query])
            
            pred = colbert.predict_lambda(Q)

            pred = pred.view(-1, 2)
            pred = torch.exp(torch.log_softmax(pred, dim=1))
            

            # if thresholding.
            pred = pred[:, 1] # take class 1 confidence
            y_pred_class = threshold(pred)

            # if argmax
            # _, y_pred_class = torch.max(pred, dim=1)

            # print(y_pred_class)
            # print(Q[0])
            # print(inference.idToText(Q[0][0]))

            pred_1_idx = torch.nonzero(y_pred_class,as_tuple=True)[0]

            
            qid = Q[0][0]
            qid = qid.to(DEVICE)
            
            pred_1_count+=len(pred_1_idx)
            total += len(torch.nonzero(Q[1],as_tuple=True)[0])

            pred_1_tokens = qid.index_select(0, pred_1_idx)
            
            tokens = inference.idToText(pred_1_tokens)
            print(tokens)
            print()

            token_id_list = list(tokens)
            for token in token_id_list:
                if token in token_count:
                    token_count[token] += 1
                else:
                    token_count[token] = 1

    print("Lambda=1 proportion: {:.3f}".format(pred_1_count/total))

    print(token_count)
    with open('./pred_1_0.5_token_count.pkl', 'wb') as f:
        pickle.dump(token_count, f)