#使用js散度加载后边训练
###nohup examples/entr/bash/zero_dropout.sh >> examples/entr/test/zero_dropout1.out 2>&1 &

base_dir=./$(dirname $0)/..
save_dir=$base_dir/checkpoints/zero_dropout
data_bin=$base_dir/data-bin
pretrained_model=$base_dir/checkpoints/baseline/checkpoint_last.pt

criterion=label_smoothed_cross_entropy
label_smoothing=0.1
dropout=0.15
lr=0.000012
lrscheduler=fixed
warmup_updates=0
max_epoch=60

echo criterion=$criterion
echo label_smoothing=$label_smoothing
echo dropout=$dropout
echo lr=$lr
echo lrscheduler=$lrscheduler
echo warmup_updates=$warmup_updates
echo max_epoch=$max_epoch
echo save_dir=$save_dir

num=1,2,3
max_tokens=4000
arch=transformer
#--warmup-init-lr 1e-07
train() {
    CUDA_VISIBLE_DEVICES=$num python train.py $data_bin \
        --save-dir $save_dir \
        --optimizer adam \
        --min-lr 1e-09 \
        --lr $lr --clip-norm 0.0 \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --lr-scheduler $lrscheduler \
        -s en -t tr \
        --dropout $dropout \
        --arch $arch \
        \
        --warmup-updates $warmup_updates \
        --weight-decay 0.0 \
        --adam-betas '(0.9, 0.98)' \
        --max-tokens $max_tokens \
        --max-epoch $max_epoch \
        \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --share-all-embeddings \
        --no-epoch-checkpoints \
        --finetune-from-model $pretrained_model
}

train
