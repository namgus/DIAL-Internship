import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

# from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher_lambda import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
from colbert.evaluation.loaders import load_queries
from colbert.evaluation.ranking import evaluate_lambda_prediction


def train(args):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if args.distributed:
        torch.cuda.manual_seed_all(12345)

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    reader = EagerBatcher(args, (0 if args.rank == -1 else args.rank), args.nranks)

    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = args.colbert # model
    colbert = colbert.to(DEVICE)
    colbert.train()
    colbert.freeze_encoder() # freeze BERT and projection layer

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)
    # colbert.parameter 중 requires_grad True인 것들만.
    # 중간 layer freeze해도 OK.
    # https://discuss.pytorch.org/t/freezing-intermediate-layers-while-training-top-and-bottom-layers/39776
    
    optimizer.zero_grad()

    amp = MixedPrecisionManager(args.amp)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    train_loss = 0.0

    start_batch_idx = 0
    
    # for debugging
    torch.set_printoptions(profile="full")

    for batch_idx, BatchSteps in zip(range(start_batch_idx, args.maxsteps), reader):
        this_batch_loss = 0.0
        correct = 0
        total = 0
        num_zeros = 0
        
        for batch in BatchSteps:
            with amp.context():
                # qid, queries, passages = batch[0], batch[1], batch[2]
                qid, queries, _ = batch[0], batch[1], batch[2]
                
                # predict_lambda에 input id랑, attention_mask 둘 다 들어가면, [batch_size, query_len] 만큼의 output이 나옴.
                pred = colbert.predict_lambda(queries)
                target = colbert.lambda_target(qid, queries[0], train=True)
                
                pred = pred.view(-1, 2)
                target = target.view(-1)

                loss = criterion(pred, target)
                loss = loss / args.accumsteps

                # train accuracy
                _, y_pred_class = torch.max(torch.log_softmax(pred, dim=1), dim=1)

                # print(y_pred_class)
                # print(target)

                num_zeros += (target==torch.zeros_like(target)).float().sum()
                correct += (y_pred_class==target).float().sum()
                total += len(target)

            amp.backward(loss)

            train_loss += loss.item()
            this_batch_loss += loss.item()

        amp.step(colbert, optimizer)
        print("Train Acc: {:.3f} / Zero proportion: {:.3f}".format(correct/total, num_zeros/total))

        if args.rank < 1:
            avg_loss = train_loss / (batch_idx+1)

            num_examples_seen = (batch_idx - start_batch_idx) * args.bsize * args.nranks
            elapsed = float(time.time() - start_time)

            log_to_mlflow = (batch_idx % 20 == 0)
            Run.log_metric('train/avg_loss', avg_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/batch_loss', this_batch_loss, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/examples', num_examples_seen, step=batch_idx, log_to_mlflow=log_to_mlflow)
            Run.log_metric('train/throughput', num_examples_seen / elapsed, step=batch_idx, log_to_mlflow=log_to_mlflow)

            print_message(batch_idx, avg_loss)
            manage_checkpoints(args, colbert, optimizer, batch_idx+1)

        if batch_idx % 1000 == 0 and batch_idx != 0:
        # if batch_idx % 1000 == 0:
            evaluate_lambda_prediction(args)
