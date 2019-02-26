#!/bin/bash -l
#SBATCH -J  ResNet
#SBATCH -o ResNet-%j.out
#SBATCH --partition Lewis,hpc5
#SBATCH --nodes=1
#SBATCH --ntasks=1         # leave at '1' unless using a MPI code
#SBATCH --cpus-per-task=5  # cores per task
#SBATCH --mem-per-cpu=10G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes

source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
module load R/R-3.3.1
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
output_dir=$gloable_dir/architecture/outputs/CNN_arch/test_out
printf "$gloable_dir\n"

python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet.py 15 5  5 'nadam' 6  50 3  $feature_dir $output_dir 5

#inter=15
#nb_filters=5
#nb_layers=5
#opt='nadam'
#filtsize='6'
#out_epoch=50
#in_epoch=3
#feature_dir = '/storage/htc/bdm/Collaboration/Zhiye/DNCON4/data/badri_training_benchmark/feats/'
#outputdir = '/scratch/jh7x3/DNCON4/architecture/CNN_arch/test_out'
#batchsize =5
