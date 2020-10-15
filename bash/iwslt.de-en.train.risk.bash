#!/bin/bash
# cp "/data1/rcduan/checkpoints/fairseq/fconv/tmp/checkpoint_last.pt" "/data1/rcduan/checkpoints/fairseq/fconv/checkpoint_last.pt"

SAVE_DIR=/data1/rcduan/checkpoints/fairseq/fconv
DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin

# --max-tokens 700 --lr 0.00005 --seq-beam 12


CUDA_VISIBLE_DEVICES=1,2,5,4 python train.py $DATABIN_DIR \
    --lr 0.0008 --min-lr 1e-09 --clip-norm 0.1 \
    --weight-decay 0 --momentum 0.99 \
    --dropout 0.3 --max-tokens 1500 \
    --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer nag \
    --task translation_struct  \
    --criterion sequence_risk --seq-beam 5 --seq-scorer bleu  


# 下边这个采用正常的crossentryloss
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py $DATABIN_DIR \
#     --lr 0.0001 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
#     --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer adam \
#      --no-epoch-checkpoints --reset-optimizer

# CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py $DATABIN_DIR \
#     --lr 0.0008 --min-lr 1e-09 --clip-norm 0.1 \
#     --weight-decay 0 --momentum 0.99 \
#     --dropout 0.3 --max-tokens 1500 \
#     --arch fconv_iwslt_de_en --save-dir $SAVE_DIR --optimizer nag \
#     --task translation_struct --reset-optimizer \
#     --criterion sequence_risk --seq-beam 5 --seq-scorer bleu  \
#      --seq-max-len-a 1.5 --seq-max-len-b 5 --no-epoch-checkpoints


# python train.py    /SISDC_GPFS/Home_SE/suda-cst/xyduan-suda/xwang/data/iwslt14.de-en/type_bin  \
#         -s de -t en  \
#         --lr 0.0008  --min-lr 1e-09 \
#         --weight-decay 0 --clip-norm 0.1  \
#         --dropout 0.3 --momentum 0.99 \
#         --max-tokens  1800   --share-all-embeddings  \
#         --arch transformer  --seq-scorer   bleu --seq-beam 6  --seq-sampling  --save-interval-updates  4000 --max-update 80000  \
#         --criterion sequence_risk     --task translation_struct  --reset-optimizer  \
#  --restore-file  /SISDC_GPFS/Home_SE/suda-cst/xyduan-suda/xwang/code/fairseq-risk/checkpoints/newtest/type-bi-share/checkpoint68.pt \
#  --save-dir  ./checkpoints/iwslt14-mrt-sap  \
