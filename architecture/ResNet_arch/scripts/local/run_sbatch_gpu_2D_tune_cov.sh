

export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
# feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
feature_dir=$gloable_dir/data/deepcov/feats/ 
output_dir=$gloable_dir/architecture/outputs/ResNet_arch/new_fea_test
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune_cov.py 150 28 6 'nadam' 3  40 1 $feature_dir $output_dir $acclog_dir 1 'VarianceScaling' 'binary_crossentropy' 0.7
# binary_crossentropy
# VarianceScaling
# lecun_normal
# RandomUniform
