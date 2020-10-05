#!/bin/bash

SAVE_DIR=/data1/rcduan/checkpoints/fairseq/fconv
DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin

CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py $DATABIN_DIR \
    --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer adam