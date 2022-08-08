import os
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run


class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        # 여기선 self.query_tokenizer pass하려고 partial 쓴거 같은데.. 근데 굳이 partial을 쓸이유가 있어?
        # 그냥 인자로 pass 하면 될거같은데. 중복해서 tokenizer를 pass할 필요가 없어서 이렇게 한건가? <- 이거인듯.
        # 암튼 self.tensorize_triples는 queries, positives, negatives, bsize를 pass하면 tensorize해주는 함수.
        # qid를 넣어서 batchsize만큼의 qid를 return 하도록 수정.

        # partial 예시
        # def power(base, exponent):
        #     return base ** exponent

        # square = partial(power, exponent=2)
        # print(square(2)) -> 4

        self.triples_path = args.triples
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        queries, positives, negatives = [], [], []
        qids = []

        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            # 여기서 이미 batch size로 잘린 triples가 들어감.

            qid, query, pos, neg = line.strip().split('\t')

            qids.append(qid)
            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        collate = self.collate(qids, queries, positives, negatives)

        return collate

    def collate(self, qids, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize
        
        # 여기서 qid도 bsize로 잘라서.
        # qid도 tensorize에 따라 들어가야 indice 순서 바뀌어 나옴.
        return self.tensorize_triples(qids, queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
