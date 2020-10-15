#!/bin/bash

DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin
PT_PATH=/data1/rcduan/checkpoints/fairseq/fconv/checkpoint339.pt

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATABIN_DIR \
    --path $PT_PATH --task translation --criterion cross_entropy \
    --batch-size 128 --beam 5 --remove-bpe --quiet

