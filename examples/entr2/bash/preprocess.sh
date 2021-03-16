base_dir=./$(dirname $0)/..
DATABIN_DIR=$base_dir/data-bin
TEXT=$base_dir/data-bin/tmp

fairseq-preprocess --source-lang en --target-lang tr \
    --trainpref $TEXT/train.bpe --validpref $TEXT/valid.bpe --testpref $TEXT/test.bpe \
    --destdir $DATABIN_DIR --joined-dictionary