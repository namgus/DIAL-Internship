python -m colbert.test --amp --doc_maxlen 180 \
--mask-punctuation \
--topk ../data/msmarco_pass/top1000.dev  \
--checkpoint ./experiments/MSMARCO-adaptation-classifier/train_termrecall.py/classifier_0_11_12/checkpoints/colbert-200000.dnn \
--root ./experiments/ --experiment MSMARCO-adaptation-classifier  --qrels ../data/msmarco_pass/qrels.dev.small.tsv \
--test_queries ../data/msmarco_pass/queries.dev.small.tsv \

# --collection ../data/msmarco_pass/collection.tsv \
# --queries ../data/msmarco_pass/queries.dev.small.tsv \
