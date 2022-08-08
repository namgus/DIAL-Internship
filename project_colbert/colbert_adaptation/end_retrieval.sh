python -m colbert.retrieve \
--amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
--queries ../data/msmarco_pass/queries.dev.small.tsv \
--nprobe 32 --faiss_depth 1024 \
--index_root ./indexes/ --index_name MSMARCO.L2.32x200k \
--checkpoint ./experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--root ./experiments/ --experiment MSMARCO-psg
