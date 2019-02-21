#!/bin/bash -l
#SBATCH -J  ResNet
#SBATCH -o ResNet-%j.out
#SBATCH --partition gpu3
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=5  # cores per task
#SBATCH --mem-per-cpu=20G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general-gpu  # investors will replace this with their account name
<<<<<<< HEAD
#SBATCH --gres gpu:"GeForce GTX 1080 Ti":1
=======
#SBATCH --gres gpu:1
>>>>>>> 26136bbf93b1d66059865e3fb3b5f5f07fa41366

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
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
<<<<<<< HEAD
output_dir=$gloable_dir/architecture/outputs/ResNet_arch/compare
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 28 6 'nadam' 3  40 1 $feature_dir $output_dir $acclog_dir 3 'VarianceScaling' 'weighted_crossentropy' 0.7

# binary_crossentropy
# VarianceScaling
# lecun_normal
# he_normal
# RandomUniform
=======
output_dir=$gloable_dir/architecture/outputs/ResNet_arch/new_fea_test
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 28  6 'nadam' 3  30 2 $feature_dir $output_dir $acclog_dir 3 'VarianceScaling' 'weighted_crossentropy' 120

>>>>>>> 26136bbf93b1d66059865e3fb3b5f5f07fa41366
