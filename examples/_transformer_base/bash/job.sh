
from_bash=remove_dirty_s.sh
save_file=reinforcement
number=0



base_dir=./$(dirname $0)/..
nohup $base_dir/bash/$from_bash >> $base_dir/test/$save_file/$save_file.$number.out 2>&1 &

