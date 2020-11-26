#generate
base_dir=./examples/_transformer_base
data_bin=$base_dir/data-bin
model=$base_dir/checkpoints/checkpoint_last.pt
beam=5

#--gen-subset来改变求bleu的对象(train, valid, test)
#自动加载test文件
CUDA_VISIBLE_DEVICES=6 python fairseq_cli/generate.py $data_bin -s ch -t en \
    --path $model --beam $beam \
    --remove-bpe --batch-size 32 --gen-subset valid --quiet


#test 28.9
#valid 15.29(best) 15.48(last)