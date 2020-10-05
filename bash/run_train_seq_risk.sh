#!/bin/bash


DATA_RAW=/data1/rcduan/data/LDC_zh-en-databin-real
#word_emb=/data/mmyin/sum_data/rush_data/independent-bpe-file/32k-bpe-inde.vec
SAVE_DIR=/data1/rcduan/checkpoints/fairseq/seq_task/base_risk/


CUDA_VISIBLE_DEVICES=4 python train.py $DATA_RAW \
 -a transformer --clip-norm 0.1 \
 --momentum 0.9 --lr 0.25 \
 --dropout 0.3 --max-tokens 200 \
 --normalize-costs \
 --seq-max-len-a 1.5 --seq-max-len-b 5 \
 --seq-sampling \
 --seq-scorer bleu \
 --criterion sequence_risk \
 --force-anneal 220 --seq-beam 12 \
 --task translation_struct --reset-optimizer \
 --save-dir $SAVE_DIR
~
