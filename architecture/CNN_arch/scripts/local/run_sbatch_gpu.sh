model_weight_out_best

export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
output_dir=$gloable_dir/architecture/outputs/CNN_arch/sample_test
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"
python $gloable_dir/architecture/CNN_arch/scripts/train_deepCNN.py 150 32 5 'nadam' 3  50 1  $feature_dir $output_dir $acclog_dir 3
