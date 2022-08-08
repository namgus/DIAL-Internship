python -m colbert.train_termrecall --amp --doc_maxlen 180 --query_maxlen 32 \
--bsize 32 --mask-punctuation \
--checkpoint ../colbert/experiments/MSMARCO-psg-split_b/train.py/msmarco.psg.rerun/checkpoints/colbert-200000.dnn \
--triples ../data/msmarco_pass/quadruples.train.small.tsv.split_b \
--root ./experiments/ --experiment MSMARCO-adaptation-classifier \
--test_queries ../data/msmarco_pass/queries.dev.small.tsv \
--lr 5e-03
