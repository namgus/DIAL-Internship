python -m colbert.test --amp --doc_maxlen 180 \
--mask-punctuation \
--topk ../data/msmarco_pass/top1000.dev  \
--checkpoint ./experiments/MSMARCO-psg-coilcolbert/train.py/msmarco.psg.berttest/checkpoints/colbert.dnn \
--root ./experiments/ --experiment MSMARCO-psg-coilcolbert  --qrels ../data/msmarco_pass/qrels.dev.small.tsv \
# --test_queries ../data/msmarco_pass/queries.dev.small.tsv

# 
# --collection ../data/msmarco_pass/collection.tsv \
# --queries ../data/msmarco_pass/queries.dev.small.tsv \

# colcoil model -> colbert score
#0.352