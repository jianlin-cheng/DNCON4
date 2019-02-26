#!/bin/bash -l
#SBATCH -J  UNet
#SBATCH -o UNet-%j.out
#SBATCH --partition gpu3
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=5  # cores per task
#SBATCH --mem-per-cpu=20G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general-gpu  # investors will replace this with their account name
#SBATCH --gres gpu:"GeForce GTX 1080 Ti":1
module load cuda/cuda-9.0.176
module load cudnn/cudnn-7.1.4-cuda-9.0.176
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
## Activate python virtual environment
# source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
source /scratch/zggc9/DNCON4_vir/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
module load R/R-3.3.1
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
# feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
feature_dir=$gloable_dir/data/deepcov/feats/ 
output_dir=$gloable_dir/architecture/outputs/UNet_arch/new_maxout3455
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture/UNet_arch/scripts/train_deepUNet_2D_gen_cov.py 320 64 2 'nadam' 5  45 1  $feature_dir $output_dir $acclog_dir 2 "glorot_uniform" "binary_crossentropy" 1

#binary_crossentropy
#weighted_crossentropy
# :"GeForce GTX 1080 Ti"