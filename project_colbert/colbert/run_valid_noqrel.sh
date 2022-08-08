python -m colbert.test --amp --doc_maxlen 180 \
--mask-punctuation \
--collection ../data/msmarco_pass/collection.tsv \
--queries ../data/msmarco_pass/queries.dev.small.tsv \
--topk ../data/msmarco_pass/top1000.dev  \
--checkpoint ./experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--root ./experiments/ --experiment MSMARCO-psg
