#!/bin/bash

DATABIN_DIR=/data1/rcduan/data/iwslt-en-ch-databin
TEXT=examples/translation/iwslt14.tokenized.de-en

fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DATABIN_DIR