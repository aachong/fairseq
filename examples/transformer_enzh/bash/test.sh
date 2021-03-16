base_dir=./$(dirname $0)/..
result=$base_dir/test/closer_all1.bleu
MODEL_PATH=$base_dir/checkpoints/closer_all1/checkpoint_last.pt

tmp=$base_dir/test/tmp
data_dir=$base_dir/data-bin
bleu=multi-bleu.perl
nist=./corpus/nist
num=2
beam=5
buffer=1024
tokens=5000


for i in 02 03 04 05 08; do

    CUDA_VISIBLE_DEVICES=$num fairseq-interactive $data_dir \
        --path $MODEL_PATH --beam $beam --buffer-size $buffer --max-tokens $tokens \
        --remove-bpe -s ch -t en --bpe fastbpe --bpe-codes $data_dir/code.ch \
        <$nist/nist$i.in >$base_dir/test/tmp/nist_$i
    grep ^H $tmp/nist_$i | cut -f3- >$tmp/nist$i

    perl $bleu -lc $nist/nist$i.ref.* <$tmp/nist$i >$tmp/nist$i.score

    echo "nist$i is over"
done

echo "checkpoint is $MODEL_PATH" >>$result
cat $tmp/nist0*.score 
cat $tmp/nist0*.score >>$result
