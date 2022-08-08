CUDA_VISIBLE_DEVICES="0,1" \
python -m torch.distributed.launch --nproc_per_node=2 -m \
colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples ../data/msmarco_pass/triples.train.small.tsv \
--root ./experiments/ --experiment MSMARCO-psg-colcoil --similarity l2 --run msmarco.psg.colcoil
