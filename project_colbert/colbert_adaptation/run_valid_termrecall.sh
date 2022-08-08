python -m colbert.test --amp --doc_maxlen 180 --query_maxlen 32 \
--mask-punctuation \
--topk ../data/msmarco_pass/top1000.dev  \
--checkpoint ./experiments/MSMARCO-oracle_lambda/train_termrecall.py/2022-02-07_04.48.54/checkpoints/colbert-50000.dnn \
--root ./experiments/ --experiment MSMARCO-psg-debug  --qrels ../data/msmarco_pass/qrels.dev.small.tsv \
# --test_queries ../data/msmarco_pass/queries.dev.small.tsv

# --collection ../data/msmarco_pass/collection.tsv \
# --queries ../data/msmarco_pass/queries.dev.small.tsv \

# --checkpoint ./experiments/MSMARCO-oracle_lambda/train_termrecall.py/2022-02-05_07.46.42/checkpoints/colbert-200000.dnn \
