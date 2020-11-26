base_dir=./examples/_transformer_big
tmp=$base_dir/test/tmp

data_bin=$base_dir/data-bin
model=$base_dir/checkpoints/checkpoint_news.pt
beam=8
CUDA_VISIBLE_DEVICES=3 fairseq-interactive $data_bin -s zh -t en \
    --path $model --beam $beam --transformer-big-zhen\
    #  --remove-bpe 
