CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=6 \
python -m torch.distributed.launch --nproc_per_node=2 -m \
colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 1024 \
--checkpoint ./experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--collection ../data/msmarco_pass/collection.tsv \
--index_root ./indexes/ --index_name MSMARCO.L2.32x200k_doc_l9 \
--root ./experiments/ --experiment MSMARCO-psg

python -m colbert.index_faiss \
--index_root ./indexes/ --index_name MSMARCO.L2.32x200k_doc_l9 \
--sample 0.3 \
--root ./experiments/ --experiment MSMARCO-psg

python -m colbert.retrieve \
--amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
--queries ../data/msmarco_pass/queries.dev.small.tsv \
--nprobe 32 --faiss_depth 1024 \
--index_root ./indexes/ --index_name MSMARCO.L2.32x200k_doc_l9 \
--checkpoint ./experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--root ./experiments/ --experiment MSMARCO-psg

