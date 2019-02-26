#!/bin/bash -l
#SBATCH -J  CnnLayernum
#SBATCH -o CnnLayernum-%j.out
#SBATCH --partition Lewis,hpc5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time 2-00:00
## Activate python virtual environment
source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/Keras1.2_TF1.5/bin/activate
module load R/R-3.3.1
GLOBAL_PATH=/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2
feature_dir=/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/features_win1_with_atch/
outputdir=/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/models/Resnet1Dconv_ss/test_results
echo "#################  Training on inter 15"
python $GLOBAL_PATH/models/Resnet1Dconv_ss/scripts/train_deepcovResnet_ss.py 15 23 5 nadam '6' 100 3 $feature_dir $outputdir
