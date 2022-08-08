python -m colbert.test --amp --doc_maxlen 180 \
--mask-punctuation \
--topk ../data/msmarco_pass/top1000.dev  \
--checkpoint ./experiments/MSMARCO-psg-colcoil/train.py/msmarco.psg.colcoil/checkpoints/colbert-200000.dnn \
--root ./experiments/ --experiment MSMARCO-psg-colcoil  --qrels ../data/msmarco_pass/qrels.dev.small.tsv \
# --test_queries ../data/msmarco_pass/queries.dev.small.tsv

# --checkpoint ./experiments/MSMARCO-psg-coilcolbert/train.py/msmarco.psg.berttest/checkpoints/colbert-300000.dnn \
# --collection ../data/msmarco_pass/collection.tsv \
# --queries ../data/msmarco_pass/queries.dev.small.tsv \

# colbert model -> colcoil score
# 0.345