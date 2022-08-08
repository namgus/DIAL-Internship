python -m colbert.index_faiss \
--index_root ./indexes/ --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 \
--root ./experiments/ --experiment MSMARCO-psg
