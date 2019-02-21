

export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
# feature_dir=$gloable_dir/data/deepcov/feats/ 
<<<<<<< HEAD
output_dir=$gloable_dir/architecture/outputs/ResNet_arch/newmaxout
=======
output_dir=$gloable_dir/architecture/outputs/ResNet_arch/new_fea_test
>>>>>>> 26136bbf93b1d66059865e3fb3b5f5f07fa41366
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

# python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 64 4 'nadam' 5  60 1  $feature_dir $output_dir $acclog_dir 1 "VarianceScaling" "weighted_crossentropy" 0.65
<<<<<<< HEAD
# python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 64 18 'nadam' 3  40 1 $feature_dir $output_dir $acclog_dir 1 'glorot_uniform' 'binary_crossentropy' 4

python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune_cov.py 150 64 6 'nadam' 3  50 1 $feature_dir $output_dir $acclog_dir 1 'he_normal' 'binary_crossentropy' 8

=======
python $gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 28 6 'nadam' 3  30 1 $feature_dir $output_dir $acclog_dir 1 'VarianceScaling' 'weighted_crossentropy' 0.7
>>>>>>> 26136bbf93b1d66059865e3fb3b5f5f07fa41366

# binary_crossentropy
# weighted_crossentropy
# VarianceScaling
# lecun_normal
# RandomUniform
