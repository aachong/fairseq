
base_dir=./examples/transformer_enzh
#BSUB -q HPC.S1.GPU.X785.sha
#BSUB -n 1
#BSUB -J DRC-job
#BSUB -gpu num=2:mode=exclusive_process
#BSUB -o $base_dir/logs/%J.out
#BSUB -e /SISDC_GPFS/Home_SE/suda-cst/xyduan-suda/rcduan/logs/%J.err

base_dir=./examples/transformer_enzh
bash $base_dir/bash/kl_train.sh >$base_dir/test/r3f_r.out
