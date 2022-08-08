import os
import time
import faiss
import random
import torch
import itertools

from colbert.utils.runs import Run
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, batch
from colbert.ranking.rankers import Ranker
from colbert.modeling.tokenization import QueryTokenizer

def retrieve(args):
    inference = ModelInference(args.colbert, amp=args.amp)
    ranker = Ranker(args, inference, faiss_depth=args.faiss_depth)

    ranking_logger = RankingLogger(Run.path, qrels=None)
    milliseconds = 0
    query_tokenizer = QueryTokenizer(32)

    with ranking_logger.context('ranking.tsv', also_save_annotations=False) as rlogger:
        queries = args.queries
        qids_in_order = list(queries.keys())

        for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]

            rankings = []

            for query_idx, q in enumerate(qbatch_text):
                torch.cuda.synchronize('cuda:0')
                s = time.time()

                Q_V = query_tokenizer.encode([q])[0] # input ids.
                # 가능하면 Q_128과 같은 크기로.

                # lambda_mask = lambda_mask(Q_V)
                Q_128 = ranker.encode([q])
                # Q_128에 맞는 사이즈의 mask를 만들고. lambda 값 query term 별로 구해놓았으니
                # 여기서 Q_128에 미리 lambda_masking을 해서 넣어버려야 함.

                pids, scores = ranker.rank(Q_128, Q_V)

                torch.cuda.synchronize()
                milliseconds += (time.time() - s) * 1000.0

                if len(pids):
                    print(qoffset+query_idx, q, len(scores), len(pids), scores[0], pids[0],
                          milliseconds / (qoffset+query_idx+1), 'ms')

                rankings.append(zip(pids, scores))

            for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
                query_idx = qoffset + query_idx

                if query_idx % 100 == 0:
                    print_message(f"#> Logging query #{query_idx} (qid {qid}) now...")

                ranking = [(score, pid, None) for pid, score in itertools.islice(ranking, args.depth)]
                rlogger.log(qid, ranking, is_ranked=True)

    print('\n\n')
    print(ranking_logger.filename)
    print("#> Done.")
    print('\n\n')


def lambda_mask(self, Q_V):
        Q_V, D_V = Q_V.to(DEVICE), D_V.to(DEVICE)
        batch_size = Q_V.shape[0]
        query_len = Q_V.shape[1]
        doc_len = D_V.shape[1]

        mask = torch.zeros(batch_size, query_len, doc_len).to(DEVICE)
        one_mask = torch.ones(1, doc_len).to(DEVICE)
        zero_mask = torch.zeros(1, doc_len).to(DEVICE)
        
        for batch_idx, query in enumerate(Q_V):
            for query_idx, query_tok in enumerate(query):
                if query_tok in D_V:
                    mask[batch_idx][query_idx] = one_mask
                else:
                    mask[batch_idx][query_idx] = zero_mask
        return mask.int().float()