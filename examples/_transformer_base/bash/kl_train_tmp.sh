#使用js散度加载后边训练

base_dir=./$(dirname $0)/..
num=0
dropout=0.3
arch=transformer
max_tokens=2000
criterion=label_smoothed_cross_entropy_r3f
label_smoothing=0.1
lrscheduler=inverse_sqrt
save_dir=$base_dir/checkpoints
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
        --max-epoch 100 \
        --no-epoch-checkpoints \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --reset-optimizer \
        --r3f-lambda 0.05 --train-subset test
    #cp out $save_dir
}

#process
train
