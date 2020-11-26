#!/bin/bash

DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin
PT_PATH=/data1/rcduan/checkpoints/fairseq/fconv/checkpoint344.pt

CUDA_VISIBLE_DEVICES=6 fairseq-generate $DATABIN_DIR \
    --path $PT_PATH --task translation --criterion sequence_risk \
    --batch-size 128 --beam 5 --remove-bpe --quiet

# 31.02,30.99,30.94,30.95,30.93,30.83,30.85,30.74,30.75,30.68
# 30.94,31.12,31.05
# 31.02,31.13,31.11
