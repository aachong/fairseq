#使用js散度加载后边训练
###nohup examples/_transformer_base/bash/kl_train.sh >> examples/_transformer_base/test/cv6.out 2>&1 &

base_dir=./$(dirname $0)/..
save_dir=$base_dir/checkpoints/reinforcement
data_bin=$base_dir/data-bin
pretrained_model=$base_dir/checkpoints/baseline/checkpoint_last.pt

criterion=cross_entropy_dirty_s
label_smoothing=0.1
dropout=0.3
lr=0.00004
lrscheduler=inverse_sqrt
warmup_updates=3000
max_epoch=210
threshold=6
extr='--warmup-init-lr 1e-07 --sensword sentence'
#--reset-optimizer

echo save_dir=$save_dir

echo criterion=$criterion
echo label_smoothing=$label_smoothing
echo dropout=$dropout
echo lr=$lr
echo lrscheduler=$lrscheduler
echo warmup_updates=$warmup_updates
echo max_epoch=$max_epoch
echo threshold=$threshold
echo extr=\'$extr\'

num=1,2,3,4
max_tokens=5000
#4400
arch=transformer
#
train() {
    CUDA_VISIBLE_DEVICES=$num python train.py $data_bin \
        --save-dir $save_dir \
        --optimizer adam \
        --min-lr 1e-09 \
        --lr $lr --clip-norm 0.0 \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --lr-scheduler $lrscheduler \
        -s ch -t en \
        --dropout $dropout \
        --arch $arch \
        --warmup-init-lr 1e-07 \
        --warmup-updates $warmup_updates \
        --weight-decay 0.0 \
        --adam-betas '(0.9, 0.98)' \
        --max-tokens $max_tokens \
        --max-epoch $max_epoch \
        \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints \
        --threshold $threshold \
        --finetune-from-model $pretrained_model \
        $extr
}

train
