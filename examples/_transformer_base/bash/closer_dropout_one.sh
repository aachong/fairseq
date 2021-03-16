#把每个decoder layer后边都计算klloss 但是每一层前边的系数是1
###nohup examples/_transformer_base/bash/closer_dropout_one.sh >> examples/_transformer_base/test/closer_one.out 2>&1 &

base_dir=./$(dirname $0)/..
save_dir=$base_dir/checkpoints/closer_one
result=$base_dir/test/closer_one.out
num=1,2,3
dropout=0.3
arch=closer_dropout
max_tokens=3024
criterion=r3f_closer_dropout_all
label_smoothing=0.1
lrscheduler=inverse_sqrt
data_bin=$base_dir/data-bin
train() {
    CUDA_VISIBLE_DEVICES=$num python train.py $data_bin \
        --optimizer adam \
        --min-lr 1e-09 \
        --lr 0.0005 --clip-norm 0.0 \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --lr-scheduler $lrscheduler \
        -s ch -t en \
        --dropout $dropout \
        --arch $arch \
        --warmup-init-lr 1e-07 \
        --warmup-updates 4000 \
        --weight-decay 0.0 \
        --adam-betas '(0.9, 0.98)' \
        --max-tokens $max_tokens \
        --save-dir $save_dir \
        --max-epoch 200 \
        \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints \
        --r3f-lambda 0.05 \
        --reset-optimizer --layer-choice allone
}


train

