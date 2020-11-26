DATABIN_DIR=/home/rcduan/fairseq/fairseq/examples/transformer_enzh/data-bin
TEXT=/home/rcduan/fairseq/fairseq/examples/transformer_enzh/data-bin/tmp

fairseq-preprocess --source-lang ch --target-lang en \
    --trainpref $TEXT/train.bpe --validpref $TEXT/valid.bpe --testpref $TEXT/test.bpe \
    --destdir $DATABIN_DIR