base_dir=./examples/_transformer_base
tmp=$base_dir/test/tmp
MODEL_PATH=$base_dir/checkpoints/tmp/checkpoint_last.pt
bleu=multi-bleu.perl
data_dir=$base_dir/data-bin
nist=./corpus/nist
num=3
beam=5
buffer=1024
tokens=3000


for i in 02 03 04 05 08; do
    # CUDA_VISIBLE_DEVICES=$num fairseq-interactive $data_dir \
    #     --path $MODEL_PATH --beam $beam --buffer-size $buffer --max-tokens $tokens \
    #     --remove-bpe -s ch -t en  \
    #     <$nist/nist$i.bpe.in >$base_dir/test/tmp/nist_$i
    #上边这个之所以下降的原因是bpe的文件应该不是和学长一样的，就是说这个就不是学长的bpe文件

    CUDA_VISIBLE_DEVICES=$num fairseq-interactive $data_dir \
        --path $MODEL_PATH --beam $beam --buffer-size $buffer --max-tokens $tokens \
        --remove-bpe -s ch -t en --bpe fastbpe --bpe-codes $data_dir/code.ch \
        <$nist/nist$i.in >$base_dir/test/tmp/nist_$i
    grep ^H $tmp/nist_$i | cut -f3- >$tmp/nist$i

    perl $bleu -lc $nist/nist$i.ref.* <$tmp/nist$i >$tmp/nist$i.score

    echo "nist$i is over"
done

echo "checkpoint is $MODEL_PATH"
cat $tmp/nist0*.score
