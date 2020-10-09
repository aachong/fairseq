#!/bin/bash

DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin
PT_PATH=/data1/rcduan/checkpoints/fairseq/fconv/checkpoint_best.pt

fairseq-generate $DATABIN_DIR \
    --path $PT_PATH \
    --batch-size 128 --beam 5 --remove-bpe --quiet