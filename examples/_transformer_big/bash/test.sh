base_dir=./examples/_transformer_big
tmp=$base_dir/test/tmp
MODEL_PATH=$base_dir/checkpoints/checkpoint_news.pt
bleu=multi-bleu.perl
data_dir=$base_dir/data-bin
nist=./corpus/nist
num=1
beam=8
buffer=1024
tokens=4000


for i in 02 03 04 05 08; do
    CUDA_VISIBLE_DEVICES=$num fairseq-interactive $data_dir \
        --path $MODEL_PATH --beam $beam --buffer-size $buffer --max-tokens $tokens \
        --remove-bpe -s zh -t en --bpe fastbpe --bpe-codes $data_dir/codes.zh \
        <$nist/nist$i.in >$base_dir/test/tmp/nist_$i
    grep ^H $tmp/nist_$i | cut -f3- >$tmp/nist$i

    perl $bleu -lc $nist/nist$i.ref.* <$tmp/nist$i >$tmp/nist$i.score

    echo "nist$i is over"
done

echo "checkpoint is $MODEL_PATH"
cat $tmp/nist0*.score >$tmp/../test2.score
