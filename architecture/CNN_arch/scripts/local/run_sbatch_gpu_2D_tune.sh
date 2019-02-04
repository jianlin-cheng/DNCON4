
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
# feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
feature_dir=$gloable_dir/data/deepcov/feats/ 
output_dir=$gloable_dir/architecture/outputs/CNN_arch/new_fea_test_gen
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

#dncon2 data
# python $gloable_dir/architecture/CNN_arch/scripts/train_deepCNN_2D_gen_tune.py 150 64 2 'nadam' 5  30 1  $feature_dir $output_dir $acclog_dir 1 "glorot_uniform" "weighted_crossentropy" 0.7 "relu"
#deepcov data
python $gloable_dir/architecture/CNN_arch/scripts/train_deepCNN_2D_gen_cov.py 150 64 2 'nadam' 5  50 1  $feature_dir $output_dir $acclog_dir 1 "glorot_uniform" "binary_crossentropy" 5 "relu"


# python $gloable_dir/architecture/CNN_arch/scripts/train_deepCNN_2D_gen_tune.py 150 64 10 'nadam' 5  30 1  $feature_dir $output_dir $acclog_dir 1 "glorot_uniform" "binary_crossentropy" 60 "relu"
# binary_crossentropy
# VarianceScaling
# lecun_normal
# RandomUniform