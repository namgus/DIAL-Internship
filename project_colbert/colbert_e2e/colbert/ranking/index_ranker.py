import os
import math
import torch
import ujson
import traceback

from itertools import accumulate
from colbert.parameters import DEVICE
from colbert.utils.utils import print_message, dotdict, flatten

BSIZE = 1 << 14


class IndexRanker():
    def __init__(self, tensor, doclens):
        self.tensor = tensor
        self.doclens = doclens

        self.maxsim_dtype = torch.float32
        self.doclens_pfxsum = [0] + list(accumulate(self.doclens)) # doclens sum.
        # e.g.) itertools.accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # [0] + list(itertools.accumulate([1,2,3,4,5])) --> [0, 1, 3, 6, 10, 15]

        self.doclens = torch.tensor(self.doclens)
        self.doclens_pfxsum = torch.tensor(self.doclens_pfxsum)

        self.dim = self.tensor.size(-1)

        self.strides = [torch_percentile(self.doclens, p) for p in [90]]
        self.strides.append(self.doclens.max().item())
        self.strides = sorted(list(set(self.strides)))

        print_message(f"#> Using strides {self.strides}..")

        self.views = self._create_views(self.tensor)
        self.buffers = self._create_buffers(BSIZE, self.tensor.dtype, {'cpu', 'cuda:0'})

    def _create_views(self, tensor):
        views = []

        for stride in self.strides:
            outdim = tensor.size(0) - stride + 1
            view = torch.as_strided(tensor, (outdim, stride, self.dim), (self.dim, self.dim, 1))
            views.append(view)

        return views

    def _create_buffers(self, max_bsize, dtype, devices):
        buffers = {}

        for device in devices:
            buffers[device] = [torch.zeros(max_bsize, stride, self.dim, dtype=dtype,
                                           device=device, pin_memory=(device == 'cpu'))
                               for stride in self.strides]

        return buffers

    def rank(self, Q, Q_V, pids, views=None, shift=0):
        assert len(pids) > 0
        assert Q.size(0) in [1, len(pids)]

        Q = Q.contiguous().to(DEVICE).to(dtype=self.maxsim_dtype)

        views = self.views if views is None else views
        VIEWS_DEVICE = views[0].device

        D_buffers = self.buffers[str(VIEWS_DEVICE)]

        raw_pids = pids if type(pids) is list else pids.tolist()
        pids = torch.tensor(pids) if type(pids) is list else pids

        doclens, offsets = self.doclens[pids], self.doclens_pfxsum[pids] # pid에 따라 doclens가 다르니까. pfxsum은 뭐지? prefix sum?

        assignments = (doclens.unsqueeze(1) > torch.tensor(self.strides).unsqueeze(0) + 1e-6).sum(-1)

        one_to_n = torch.arange(len(raw_pids))
        output_pids, output_scores, output_permutation = [], [], [] # list

        for group_idx, stride in enumerate(self.strides):
            locator = (assignments == group_idx)  # locator라는게 뭘 하는 걸까나.. 인덱싱 용돈데.

            if locator.sum() < 1e-5:
                continue

            group_pids, group_doclens, group_offsets = pids[locator], doclens[locator], offsets[locator]
            group_Q = Q if Q.size(0) == 1 else Q[locator] # Q.size(0) == 1. 그래서 그냥 Q.

            group_offsets = group_offsets.to(VIEWS_DEVICE) - shift
            group_offsets_uniq, group_offsets_expand = torch.unique_consecutive(group_offsets, return_inverse=True)

            D_size = group_offsets_uniq.size(0)
            D = torch.index_select(views[group_idx], 0, group_offsets_uniq, out=D_buffers[group_idx][:D_size])
            D = D.to(DEVICE)
            D = D[group_offsets_expand.to(DEVICE)].to(dtype=self.maxsim_dtype)

            # print("D.shape")
            # [batch_size, doc_len, token_dim]
            # [random, 180 or 108, 128]
            # 근데 왜 batch_size랑 doc_len이 계속 바뀜? iteration 마다?
            # doc len은 180이랑 108 섞여 나오고.
            # batch_size는 완전 중구난방이네.

            # print("group_Q.shape")
            # [batch, token_dim, query_len]
            # [1, 128, 32]

            mask = torch.arange(stride, device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= group_doclens.to(DEVICE).unsqueeze(-1)

            scores = (D @ group_Q) * mask.unsqueeze(-1)
            # print(scores.shape)
            # [batch_size, doc_len, query_len]
            # [random, 180 or 108, 32]
            
            scores = scores.permute(0, 2, 1)
            # scores.shape
            # [batch_size, query_len, doc_len]

            # mask도 똑같이.
            # batch별로, query token이 doc내에 있는지 CHECK하는 coil_MASK
            # lambda에 따라 query token에 따른 줄 단위 mask.
            
            # lambda를 곱해주려면 vocab의 token_id가 필요한데.
            scores = scores.max(1).values.sum(-1).cpu()

            output_pids.append(group_pids)
            output_scores.append(scores)
            output_permutation.append(one_to_n[locator])

        output_permutation = torch.cat(output_permutation).sort().indices
        output_pids = torch.cat(output_pids)[output_permutation].tolist() # list of tensor들을 concat 후 tolist?
        output_scores = torch.cat(output_scores)[output_permutation].tolist() # list of tensor들을 concat후 tolist?

        assert len(raw_pids) == len(output_pids)
        assert len(raw_pids) == len(output_scores)
        assert raw_pids == output_pids

        return output_scores

    def batch_rank(self, all_query_embeddings, all_query_indexes, all_pids, sorted_pids):
        assert sorted_pids is True

        scores = []
        range_start, range_end = 0, 0

        for pid_offset in range(0, len(self.doclens), 50_000):
            pid_endpos = min(pid_offset + 50_000, len(self.doclens))

            range_start = range_start + (all_pids[range_start:] < pid_offset).sum()
            range_end = range_end + (all_pids[range_end:] < pid_endpos).sum()

            pids = all_pids[range_start:range_end]
            query_indexes = all_query_indexes[range_start:range_end]

            print_message(f"###--> Got {len(pids)} query--passage pairs in this sub-range {(pid_offset, pid_endpos)}.")

            if len(pids) == 0:
                continue

            print_message(f"###--> Ranking in batches the pairs #{range_start} through #{range_end} in this sub-range.")

            tensor_offset = self.doclens_pfxsum[pid_offset].item()
            tensor_endpos = self.doclens_pfxsum[pid_endpos].item() + 512

            collection = self.tensor[tensor_offset:tensor_endpos].to(DEVICE)
            views = self._create_views(collection)

            print_message(f"#> Ranking in batches of {BSIZE} query--passage pairs...")

            for batch_idx, offset in enumerate(range(0, len(pids), BSIZE)):
                if batch_idx % 100 == 0:
                    print_message("#> Processing batch #{}..".format(batch_idx))

                endpos = offset + BSIZE
                batch_query_index, batch_pids = query_indexes[offset:endpos], pids[offset:endpos]

                Q = all_query_embeddings[batch_query_index]

                scores.extend(self.rank(Q, batch_pids, views, shift=tensor_offset))

        return scores


def torch_percentile(tensor, p):
    assert p in range(1, 100+1)
    assert tensor.dim() == 1

    return tensor.kthvalue(int(p * tensor.size(0) / 100.0)).values.item()
