import os
import random
import pandas as pd
import pickle

import colbert.utils.distributed as distributed

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.training.training_termrecall import train
from colbert.evaluation.loaders import load_colbert
from colbert.evaluation.loaders import load_queries


def main():
    random.seed(12345)

    parser = Arguments(description='Exhaustive (slow, not index-based) evaluation of re-ranking with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_training_parameters_termrecall()

    parser.add_argument('--depth', dest='depth', required=False, default=None, type=int)

    args = parser.parse()

    assert args.bsize % args.accumsteps == 0, ((args.bsize, args.accumsteps),
                                               "The batch size must be divisible by the number of gradient accumulation steps.")
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    args.colbert, args.checkpoint = load_colbert(args)
    args.test_queries = load_queries(args.test_queries)

    with Run.context(consider_failed_if_interrupted=False):
        train(args) # from training_termrecall

    # with Run.context():
    #     args.colbert, args.checkpoint = load_colbert(args)

    #     # with open('../data/msmarco_pass/oracle_lambda_dev.pkl', 'rb') as f:
    #     #     args.colbert.lambda_oracle = pickle.load(f)

    #     args.qrels = load_qrels(args.qrels)

    #     evaluate_recall(args.qrels, args.queries, args.topK_pids)
    #     evaluate(args)


if __name__ == "__main__":
    main()
