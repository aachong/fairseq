###nohup examples/entr/bash/kl_train.sh >> examples/entr/test/closer_gap1.out 2>&1 &

from_bash=train.sh
save_file=drc_residual_norm
number=1



base_dir=./$(dirname $0)/..
nohup $base_dir/bash/$from_bash > $base_dir/test/$save_file/$save_file.$number.out 2>&1 &

