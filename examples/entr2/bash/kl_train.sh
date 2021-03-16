#使用js散度加载后边训练
###nohup examples/entr/bash/kl_train.sh >> examples/entr/test/closer_gap1.out 2>&1 &


base_dir=./$(dirname $0)/..
save_dir=$base_dir/checkpoints/closer-all
data_bin=$base_dir/data-bin
pretrained_model=$base_dir/checkpoints/baseline/checkpoint_last.pt

# criterion=label_smoothed_cross_entropy
criterion=label_smoothed_cross_entropy_r3f_noised_input
label_smoothing=0.1
dropout=0.3
lr=0.00004
lrscheduler=inverse_sqrt
warmup_updates=3000
max_epoch=201
r3f_lambda=1
extr='--warmup-init-lr 1e-07 --reset-optimizer'
#--noised-no-grad --noised-eval-model --eps 1e2 --self-training-drc --cv 
#--de-std --reset-optimizer
echo save_dir=$save_dir

echo 在最后一层正则化之前输出，模型是baseline_continue #information
echo criterion=$criterion
echo label_smoothing=$label_smoothing
echo dropout=$dropout
echo lr=$lr
echo lrscheduler=$lrscheduler
echo warmup_updates=$warmup_updates
echo max_epoch=$max_epoch
echo r3f_lambda=$r3f_lambda
echo extr=\'$extr\'

num=0
max_tokens=4000
#2梯度 4000 1 梯度 10000 
arch=transformer
#closer_dropout
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
        --r3f-lambda $r3f_lambda \
        $extr
}

train
#--finetune-from-model $pretrained_model \