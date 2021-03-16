#generate
base_dir=./$(dirname $0)/..
data_bin=$base_dir/data-bin
model=$base_dir/checkpoints/closer-all/checkpoint_best.pt
beam=5

#--gen-subset来改变求bleu的对象(train, valid, test)
#自动加载test文件
CUDA_VISIBLE_DEVICES=3 python fairseq_cli/generate.py $data_bin -s en -t tr \
    --path $model --beam $beam \
    --remove-bpe sentencepiece --max-tokens 15000

#test 28.9
#valid 15.29(best) 15.48(last)

# fn=r3f1
# grep ^S $fn.out | cut -f2- >$fn.en
# grep ^T $fn.out | cut -f2- >$fn.tr
# grep ^H $fn.out | cut -f3- >$fn.h.tr
# compare-mt --output_directory output/ example/ted.ref.eng example/ted.sys1.eng example/ted.sys2.eng