

export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
output_dir=$gloable_dir/architecture/outputs/CRMN_arch/test_out
printf "$gloable_dir\n"

python $gloable_dir/architecture/CRMN_arch/scripts/train_deepCRMN.py 150 35  2 'nadam' 6  50 1  $feature_dir $output_dir 1
