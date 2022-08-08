python -m colbert.test --amp --doc_maxlen 180 \
--mask-punctuation \
--topk ../data/msmarco_pass/top1000.dev  \
--checkpoint ./experiments/MSMARCO-oracle_lambda/train_termrecall.py/colbert_classifier_train_90per/checkpoints/colbert-200000.dnn \
--root ./experiments/ --experiment MSMARCO-oracle_lambda  --qrels ../data/msmarco_pass/qrels.dev.small.tsv \
--test_queries ../data/msmarco_pass/queries.dev.small.tsv \

# --collection ../data/msmarco_pass/collection.tsv \
# --queries ../data/msmarco_pass/queries.dev.small.tsv \
