#!/bin/bash -l
#SBATCH -J  test
#SBATCH -o test-%j.out
#SBATCH --partition gpu3
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=5  # cores per task
#SBATCH --mem-per-cpu=20G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes
#SBATCH --qos=normal
#SBATCH --account=general-gpu  # investors will replace this with their account name
#SBATCH --gres gpu:1
module load cuda/cuda-8.0
export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=""
## Activate python virtual environment
source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
module load R/R-3.3.1
feature_dir=/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/
outputdir=/scratch/jh7x3/DNCON4/architecture/CNN_arch/test_out2
echo "#################  Training on inter 15"
THEANO_FLAGS=floatX=float32,device=gpu python /storage/htc/bdm/Collaboration/Zhiye/DNCON4/architecture/CNN_arch/scripts/train_deepCNN.py 15 5  5 'nadam' 6  50 3  $feature_dir $outputdir 5


#inter=15
#nb_filters=10
#nb_layers=10
#opt='nadam'
#filtsize='6'
#out_epoch=1
#in_epoch=1
#feature_dir = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
#outputdir = '/scratch/jh7x3/DNCON4/architecture/CNN_arch/test_out'
#batchsize =5
