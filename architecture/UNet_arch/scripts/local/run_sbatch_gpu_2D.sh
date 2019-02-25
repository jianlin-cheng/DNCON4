
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
# feature_dir=$gloable_dir/data/badri_training_benchmark/feats_dncon2/
output_dir=$gloable_dir/architecture/outputs/CNN_arch/new_fea_test_gen
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture/CNN_arch/scripts/train_deepCNN_2D_gen.py 150 32 5 'nadam' 5  20 3  $feature_dir $output_dir $acclog_dir 1
