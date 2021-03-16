base_dir=./examples/_transformer_base

data_bin=$base_dir/data-bin
pt_path=$base_dir/checkpoints/baseline/checkpoint_last.pt
nist=./corpus/nist

CUDA_VISIBLE_DEVICES=4 fairseq-interactive \
    --path $pt_path $data_bin --source-lang ch --target-lang en \
    --beam 5 --remove-bpe \
    --tokenizer moses \
    --bpe fastbpe --bpe-codes $data_bin/code.en
