fastbpe=/home/rcduan/fairseq/fairseq/examples/tutorial/fastbpe/fastBPE
base_dir=./$(dirname $0)/..
save_bin=$base_dir/data-bin/tmp
corpus=/home/rcduan/fairseq/fairseq/corpus/entr
s=tr
t=en
$fastbpe/fast learnbpe 20000 $corpus/train.$s >$save_bin/codes.$s
$fastbpe/fast learnbpe 20000 $corpus/train.$t >$save_bin/codes.$t
$fastbpe/fast applybpe $save_bin/train.bpe.$s $corpus/train.$s $save_bin/codes.$s
$fastbpe/fast applybpe $save_bin/train.bpe.$t $corpus/train.$t $save_bin/codes.$t

$fastbpe/fast getvocab $save_bin/train.bpe.$s >$save_bin/vocab.$s
$fastbpe/fast getvocab $save_bin/train.bpe.$t >$save_bin/vocab.$t

$fastbpe/fast applybpe $save_bin/valid.bpe.$s $corpus/valid.$s $save_bin/codes.$s $save_bin/vocab.$s
$fastbpe/fast applybpe $save_bin/valid.bpe.$t $corpus/valid.$t $save_bin/codes.$t $save_bin/vocab.$t
$fastbpe/fast applybpe $save_bin/test.bpe.$s $corpus/test.$s $save_bin/codes.$s $save_bin/vocab.$s
$fastbpe/fast applybpe $save_bin/test.bpe.$t $corpus/test.$t $save_bin/codes.$t $save_bin/vocab.$t
