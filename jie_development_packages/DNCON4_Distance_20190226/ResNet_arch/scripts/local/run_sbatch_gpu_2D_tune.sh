

export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
# feature_dir=$gloable_dir/data/deepcov/feats/ 
output_dir=$gloable_dir/architecture_distance/outputs/ResNet_arch/newmaxout
acclog_dir=$gloable_dir/architecture_distance/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture_distance/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 64 6 'nadam' 3  200 1 $feature_dir $output_dir $acclog_dir 1 'he_normal' 'weighted_MSE' 18

# binary_crossentropy
# weighted_crossentropy
# VarianceScaling
# lecun_normal
# RandomUniform