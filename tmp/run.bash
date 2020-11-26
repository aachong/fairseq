MODEL_DIR=/data1/rcduan/wmt14.en-fr.fconv-py/

fairseq-interactive \
    --path $MODEL_DIR/model.pt $MODEL_DIR \
    --beam 5 --source-lang en --target-lang fr \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes $MODEL_DIR/bpecodes