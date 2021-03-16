#使用js散度加载后边训练
###nohup examples/_transformer_base/bash/kl_train.sh >> examples/_transformer_base/test/cv6.out 2>&1 &

base_dir=./$(dirname $0)/..
save_dir=$base_dir/checkpoints/baseline
data_bin=$base_dir/data-bin
pretrained_model=$base_dir/checkpoints/closer_all/checkpoint_1705.pt

criterion=label_smoothed_cross_entropy_r3f_noised_input
label_smoothing=0.1
dropout=0.3
lr=0.00004
lrscheduler=inverse_sqrt
warmup_updates=4000
max_epoch=200
r3f_lambda=0
extr='--warmup-init-lr 1e-07 --reset-optimizer'
#--noised-no-grad --noised-eval-model --eps 1e2 --self-training-drc --cv

echo save_dir=$save_dir

echo criterion=$criterion
echo label_smoothing=$label_smoothing
echo dropout=$dropout
echo lr=$lr
echo lrscheduler=$lrscheduler
echo warmup_updates=$warmup_updates
echo max_epoch=$max_epoch
echo r3f_lambda=$r3f_lambda
echo extr=\'$extr\'

num=5
max_tokens=3200
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
        --r3f-lambda $r3f_lambda \
        $extr
        # --finetune-from-model $pretrained_model \
}

train
