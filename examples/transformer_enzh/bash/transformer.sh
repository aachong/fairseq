num=0,1
dropout=0.3
arch=transformer
max_tokens=4096
criterion=label_smoothed_cross_entropy_r3f
label_smoothing=0.1
lrscheduler=inverse_sqrt
save_dir=./checkpoints
data_bin=../data-bin
train(){
CUDA_VISIBLE_DEVICES=$num python train.py $data_bin \
                        --optimizer adam \
                        --min-lr  1e-09 \
                        --lr  0.0005\
                        --clip-norm 0.0 \
                        --criterion $criterion \
                        --label-smoothing $label_smoothing \
                        --lr-scheduler $lrscheduler \
			            -s zh -t en \
                        --dropout $dropout \
                        --arch $arch \
                        --warmup-init-lr 1e-07 \
                        --warmup-updates 4000 \
                        --weight-decay 0.0 \
                        --adam-betas '(0.9, 0.98)' \
                        --max-tokens $max_tokens \
                        --save-dir $save_dir \
                        --max-epoch 100 --fp16 \
                        --no-epoch-checkpoints \
                        --eval-bleu  --eval-bleu-remove-bpe \
                        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                        --r3f-lambda 0.05

#cp out $save_dir
}

#process
train
