#!/bin/bash -l
#SBATCH -J  Filtersize_evalu
#SBATCH -o Filtersize_evalu-%j.out
#SBATCH --partition Lewis,hpc5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time 2-00:00
## Activate python virtual environment
source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/Keras1.2_TF1.5/bin/activate
module load R/R-3.3.1
export HDF5_USE_FILE_LOCKING=FALSE
GLOBAL_PATH=/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2
feature_dir=/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/features_win1_with_atch/
output_dir=/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/models/AttConv_ss/filternum_results
acclog_dir=/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/models/AttConv_ss/filternum_evalu_results
python $GLOBAL_PATH/lib/test_dnss.py $GLOBAL_PATH/datasets/dncov_training.list  15 5 5 nadam '6' $feature_dir $output_dir $acclog_dir 'attention_cov'
#python $GLOBAL_PATH/lib/test_dnss.py $GLOBAL_PATH/datasets/dncov_validation.list  15 5 5 nadam '6' $feature_dir $output_dir $acclog_dir 'attention_cov'
#python $GLOBAL_PATH/lib/test_dnss.py $GLOBAL_PATH/datasets/adj_dncon-test.lst  15 5 5 nadam '6' $feature_dir $output_dir $acclog_dir 'attention_cov'
