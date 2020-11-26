fastbpe=/home/rcduan/fairseq/fairseq/examples/tutorial/fastbpe/fastBPE
save_bin=/home/rcduan/fairseq/fairseq/examples/transformer_enzh/data-bin/tmp
corpus=/home/rcduan/fairseq/fairseq/corpus/ldc.enzh

$fastbpe/fast learnbpe 20000 $corpus/20w.train.ch >$save_bin/codes.ch
$fastbpe/fast learnbpe 20000 $corpus/20w.train.en >$save_bin/codes.en
$fastbpe/fast applybpe $save_bin/train.bpe.ch $corpus/20w.train.ch $save_bin/codes.ch
$fastbpe/fast applybpe $save_bin/train.bpe.en $corpus/20w.train.en $save_bin/codes.en

$fastbpe/fast getvocab $save_bin/train.bpe.ch >$save_bin/vocab.ch
$fastbpe/fast getvocab $save_bin/train.bpe.en >$save_bin/vocab.en

$fastbpe/fast applybpe $save_bin/valid.bpe.ch $corpus/valid.ch $save_bin/codes.ch $save_bin/vocab.ch
$fastbpe/fast applybpe $save_bin/valid.bpe.en $corpus/valid.en $save_bin/codes.en $save_bin/vocab.en
$fastbpe/fast applybpe $save_bin/test.bpe.ch $corpus/test.ch $save_bin/codes.ch $save_bin/vocab.ch
$fastbpe/fast applybpe $save_bin/test.bpe.en $corpus/test.en $save_bin/codes.en $save_bin/vocab.en
