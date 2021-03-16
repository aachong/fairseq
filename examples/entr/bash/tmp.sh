CUDA_VISIBLE_DEVICES=$num python train.py $data_bin \
                        --optimizer adam --min-lr  1e-09 --lr  0.0005 --clip-norm 0.0 \
                        --criterion $criterion --label-smoothing $label_smoothing \
                        --lr-scheduler $lrscheduler -s $src -t $tgt --dropout $dropout \
                        --arch $arch \
                        --warmup-init-lr 1e-07 --warmup-updates 4000 --weight-decay 0.0 --adam-betas '(0.9, 0.98)' \
                        --max-tokens $max_tokens \
                        --save-dir $save_dir \
                        --max-epoch 100 \
                        --share-all-embeddings \
                        --eval-bleu  --eval-bleu-remove-bpe sentencepiece \
                        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                        --no-epoch-checkpoints   \
                        --ddp-backend no_c10d
}
train