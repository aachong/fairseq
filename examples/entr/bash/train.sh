#正常训练文件
#nohup examples/entr/bash/train.sh >> examples/entr/test/baseline.out 2>&1 &

base_dir=./$(dirname $0)/..
num=2,3
dropout=0.3
arch=transformer
max_tokens=7000
criterion=label_smoothed_cross_entropy
label_smoothing=0.1
lrscheduler=inverse_sqrt
save_dir=$base_dir/checkpoints/baseline
data_bin=$base_dir/data-bin
train() {
    CUDA_VISIBLE_DEVICES=$num python train.py $data_bin \
        --optimizer adam \
        --min-lr 1e-09 \
        --lr 0.0005 --clip-norm 0.0 \
        --criterion $criterion \
        --label-smoothing $label_smoothing \
        --lr-scheduler $lrscheduler \
        -s en -t tr \
        --dropout $dropout \
        --arch $arch \
        --warmup-init-lr 1e-07 \
        --warmup-updates 4000 \
        --weight-decay 0.0 \
        --adam-betas '(0.9, 0.98)' \
        --max-tokens $max_tokens \
        --save-dir $save_dir \
        --max-epoch 250 \
        \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints \
        --share-all-embeddings \
        # --reset-optimizer

}

#process
train
