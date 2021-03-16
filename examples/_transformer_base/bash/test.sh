base_dir=./$(dirname $0)/..
result=$base_dir/test/closer_all/closer_all.0.bleu
MODEL_PATH=$base_dir/checkpoints/closer_all/checkpoint_best.pt
task_name=170

num=1,3
tmp=$base_dir/test/tmp
bleu=multi-bleu.perl
data_dir=$base_dir/data-bin
nist=./corpus/nist
beam=8
buffer=1024
tokens=8000


for i in 02 03 04 05 08; do

    CUDA_VISIBLE_DEVICES=$num fairseq-interactive $data_dir \
        --path $MODEL_PATH --beam $beam --buffer-size $buffer --max-tokens $tokens \
        --remove-bpe -s ch -t en --bpe fastbpe --bpe-codes $data_dir/code.ch \
        <$nist/nist$i.in >$base_dir/test/tmp/nist_$i
    grep ^H $tmp/nist_$i | cut -f3- >$tmp/nist$i

    perl $bleu -lc $nist/nist$i.ref.* <$tmp/nist$i >$tmp/nist$i.score

    echo "nist$i is over"
done

show(){
    echo "----------------------------------------------------------------"
    echo "results are from $task_name" 
    cat $tmp/../$task_name | grep 'end of epoch ...' -o | tail -n 1
    echo "checkpoint is $MODEL_PATH" 
    cat $tmp/nist0*.score 
    cat $tmp/nist0*.score | grep 'BLEU = .....' -o | awk '{sum+=$3} END {print "Avg =", sum/NR}'
}

show >>$result
cat $tmp/nist0*.score 
