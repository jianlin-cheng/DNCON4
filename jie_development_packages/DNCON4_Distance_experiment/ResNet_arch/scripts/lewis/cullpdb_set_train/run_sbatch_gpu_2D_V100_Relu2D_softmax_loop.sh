#!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --job-name=cRN-soft         # name for the job
#SBATCH -o ResNet-soft-cullpdb-%j.out
#SBATCH --cpus-per-task=1                   # number of cores
#SBATCH --mem=40G                           # total memory
#SBATCH --time 00-2:00:00                   # time limit in the form days-hours:minutes

#SBATCH --partition Gpu
#SBATCH --gres gpu:"Tesla V100-PCIE-32GB":1
#--------------------------------------------------------------------------------

module load cuda/cuda-9.0.176
module load cudnn/cudnn-7.1.4-cuda-9.0.176
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
## Activate python virtual environment
source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
#source /scratch/zggc9/DNCON4_vir/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
module load R/R-3.3.1
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
#feature_dir=/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/
feature_dir=/storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output/
output_dir=/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/Test/cullpdb_Distance_Relu2D_softmax_inloop
acclog_dir=/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/Test/cullpdb_Distance_Relu2D_softmax_inloop
printf "$gloable_dir\n"

#python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 64 6 'nadam' 3  40 1 $feature_dir $output_dir $acclog_dir 1 'he_normal' 'binary_crossentropy' 1
python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/train_deepResNet_2D_gen_tune_cullpdb.py 150 64 6 'nadam' 3  100 1 $feature_dir $output_dir $acclog_dir 1 'he_normal' 'categorical_crossentropy' 1


# binary_crossentropy
# VarianceScaling
# lecun_normal
# he_normal
# RandomUniform

#"GeForce GTX 1080 Ti":1
