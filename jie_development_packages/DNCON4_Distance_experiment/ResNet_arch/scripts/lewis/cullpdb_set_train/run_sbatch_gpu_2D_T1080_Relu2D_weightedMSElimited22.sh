#!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --job-name=cRN-wMSEli         # name for the job
#SBATCH -o ResNet-wMSEli22-cullpdb-T1080-%j.out
#SBATCH --partition gpu3
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=2  # cores per task
#SBATCH --mem-per-cpu=10G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general-gpu  # investors will replace this with their account name
#SBATCH --gres gpu:"GeForce GTX 1080 Ti":1
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
output_dir=/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/Test/cullpdb_Distance_Relu2D_weighted_MSElimited22_inloop
acclog_dir=/scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/Test/cullpdb_Distance_Relu2D_weighted_MSElimited22_inloop
printf "$gloable_dir\n"

#python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 64 6 'nadam' 3  40 1 $feature_dir $output_dir $acclog_dir 1 'he_normal' 'binary_crossentropy' 1
python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/train_deepResNet_2D_gen_tune_cullpdb.py 150 64 6 'nadam' 3  100 1 $feature_dir $output_dir $acclog_dir 1 'he_normal' 'weighted_MSE_limited22' 1


# binary_crossentropy
# VarianceScaling
# lecun_normal
# he_normal
# RandomUniform

#"GeForce GTX 1080 Ti":1
#Tesla K40m
