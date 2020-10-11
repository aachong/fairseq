#!/bin/bash

DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin
PT_PATH=/data1/rcduan/checkpoints/fairseq/fconv/checkpoint_best.pt

# wie würden sie besser sein wollen , als sie momentan sind ?
# how would you like to be better than you are ?

fairseq-interactive   \
    --path $PT_PATH $DATABIN_DIR \
    --beam 12 --source-lang de --target-lang en \
    --tokenizer moses --task translation_struct \
    --bpe subword_nmt --bpe-codes $DATABIN_DIR/bpecodes --sampling --nbest 12