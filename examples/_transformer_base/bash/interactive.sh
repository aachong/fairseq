base_dir=./examples/_transformer_base

data_bin=$base_dir/data-bin
pt_path=$base_dir/checkpoints/tmp/checkpoint_last.pt
nist=./corpus/nist

CUDA_VISIBLE_DEVICES=2 fairseq-interactive \
    --path $pt_path $data_bin --source-lang ch --target-lang en \
    --beam 5 --remove-bpe \
    # --tokenizer moses --max-tokens 2000  --buffer-size 2000 \
    # --bpe subword_nmt --bpe-codes $data_bin/code.ch <$nist/nist08.in >$base_dir/test/nist_08
