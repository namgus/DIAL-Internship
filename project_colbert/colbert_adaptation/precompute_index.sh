CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=6 \
python -m torch.distributed.launch --nproc_per_node=2 -m \
colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--checkpoint ./experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--collection ../data/msmarco_pass/collection.tsv \
--index_root ./indexes/ --index_name MSMARCO.L2.32x200k_doc_l6 \
--root ./experiments/ --experiment MSMARCO-psg
