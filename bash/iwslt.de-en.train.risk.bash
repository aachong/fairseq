#!/bin/bash
cp "/data1/rcduan/checkpoints/fairseq/fconv/tmp/checkpoint_last.pt" "/data1/rcduan/checkpoints/fairseq/fconv/checkpoint_last.pt"

SAVE_DIR=/data1/rcduan/checkpoints/fairseq/fconv
DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin

# --max-tokens 700 --lr 0.00005 --seq-beam 12


CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py $DATABIN_DIR \
    --lr 0.00001 --clip-norm 0.1 --dropout 0.3 --max-tokens 1500 \
    --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer adam \
    --task translation_struct --reset-optimizer \
    --criterion sequence_risk --seq-beam 4 --seq-scorer bleu  \
     --seq-max-len-a 1.5 --seq-max-len-b 5 --no-epoch-checkpoints

# 下边这个采用正常的crossentryloss
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py $DATABIN_DIR \
#     --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
#     --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer adam \
#      --no-epoch-checkpoints --reset-optimizer





