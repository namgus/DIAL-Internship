import torch

from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.parameters import DEVICE


class ModelInference():
    def __init__(self, colbert: ColBERT, amp=False):
        assert colbert.training is False

        self.colbert = colbert
        # self.query_tokenizer = QueryTokenizer(colbert.query_maxlen)
        # self.doc_tokenizer = DocTokenizer(colbert.doc_maxlen)
        self.query_tokenizer = QueryTokenizer(32)
        self.doc_tokenizer = DocTokenizer(180)

        self.amp_manager = MixedPrecisionManager(amp)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc(*args, **kw_args)
                return D.cpu() if to_cpu else D

    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            input_ids_batch = [input_ids for input_ids, attention_mask in batches]
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            # return torch.cat(input_ids_batch), torch.cat(batches)
            return batches

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
        return input_ids, attention_mask
        # return input_ids, self.query(input_ids, attention_mask)
    
    def idToText(self, token_id):
        return list(self.query_tokenizer.tok.convert_ids_to_tokens(list(token_id)))

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        # if bsize:
        #     batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)
        #     input_ids_batch = [input_ids for input_ids, attention_mask in batches]
        #     batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
        #                for input_ids, attention_mask in batches]
        #     return batches

            # return torch.cat(input_ids_batch), torch.cat(batches)


            # if keep_dims:
            #     D = _stack_3D_tensors(batches)
            #     return D[reverse_indices]

            # D = [d for batch in batches for d in batch]
            # print(D.shape)

            # input_ids_batch = [input_ids_batch[idx] for idx in reverse_indices.tolist()]
            # print("input_ids_batch")
            # print(len(input_ids_batch))
            # print(input_ids_batch[0])
            # print(input_ids_batch[0].shape)
            # return input_ids_batch[0], torch.cat([D[idx] for idx in reverse_indices.tolist()])

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return input_ids, attention_mask
        return input_ids, self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    def score(self, Q, D, mask=None, lengths=None, explain=False):
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= lengths.to(DEVICE).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        if explain:
            assert False, "TODO"

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output
