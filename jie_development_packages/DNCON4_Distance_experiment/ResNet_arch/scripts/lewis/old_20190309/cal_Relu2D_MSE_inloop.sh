#!/bin/bash -l

job=$(sbatch run_sbatch_gpu_2D_V100_Relu2D_MSE_loop.sh)
jid="$(cut -d " " -f4 <<<$job)"
for i in $(seq 1 30)
do
job=$(sbatch --dependency=afterany:$jid run_sbatch_gpu_2D_V100_Relu2D_MSE_loop.sh)
jid="$(cut -d " " -f4 <<<$job)"
done
