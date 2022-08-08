python -m colbert.test --amp --doc_maxlen 180 \
--mask-punctuation \
--topk ../data/msmarco_pass/top1000.dev  \
--checkpoint ./experiments/MSMARCO-psg-colcoil/train.py/msmarco.psg.colcoil/checkpoints/colbert-200000.dnn \
--root ./experiments/ --experiment MSMARCO-psg  --qrels ../data/msmarco_pass/qrels.dev.small.tsv

# --collection ../data/msmarco_pass/collection.tsv \
# --queries ../data/msmarco_pass/queries.dev.small.tsv \