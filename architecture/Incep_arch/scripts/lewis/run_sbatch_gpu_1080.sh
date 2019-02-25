#!/bin/bash -l
#SBATCH -J  Inception
#SBATCH -o Inception-%j.out
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
source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
module load R/R-3.3.1
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
output_dir=$gloable_dir/architecture/outputs/Incep_arch/
printf "$gloable_dir\n"

python $gloable_dir/architecture/Incep_arch/scripts/train_deepIncep_allfea.py 150 32 4 'nadam' 5 50 1  $feature_dir $output_dir 25


#inter=15
#nb_filters=5
#nb_layers=5
#opt='nadam'
#filtsize='6'
#out_epoch=50
#in_epoch=3
#feature_dir = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
#output_dir = '/scratch/jh7x3/DNCON4/architecture/CNN_arch/test_out'
#batchsize =5
# Tesla P100-PCIE-12GB:1