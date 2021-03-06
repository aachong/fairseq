#! /bin/bash

SAVE_DIR=/data1/rcduan/checkpoints/fairseq/fconv/
DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin

# --max-tokens 700 --lr 0.00005 --seq-beam 12

# CUDA_VISIBLE_DEVICES=5,6,7 python train.py $DATABIN_DIR \
#     --lr 0.00005 --clip-norm 0.05 --dropout 0.2 --max-tokens 700 \
#     --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer adam \
#     --task translation_struct --reset-optimizer \
#     --criterion sequence_risk --seq-beam 12 --seq-scorer bleu  \
#      --seq-max-len-a 1.5 --seq-max-len-b 5 --no-epoch-checkpoints

# 下边这个采用正常的crossentryloss
CUDA_VISIBLE_DEVICES=6,7 python train.py $DATABIN_DIR \
    --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer adam \
      --reset-optimizer \
     --criterion label_smoothed_cross_entropy_r3f --r3f-lambda 0.01 





