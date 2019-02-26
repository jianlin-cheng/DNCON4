

export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
# feature_dir=$gloable_dir/data/badri_training_benchmark/feats_dncon2/
output_dir=$gloable_dir/architecture/outputs/Incep_arch/new_fea_test
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture/Incep_arch/scripts/train_deepIncep_2D.py 150 32 6 'nadam' 3 50 1  $feature_dir $output_dir $acclog_dir 1
