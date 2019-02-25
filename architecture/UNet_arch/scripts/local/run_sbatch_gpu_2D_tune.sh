
export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
# feature_dir=$gloable_dir/data/deepcov/feats/ 
output_dir=$gloable_dir/architecture/outputs/UNet_arch/sample120
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

#dncon2 data
python $gloable_dir/architecture/UNet_arch/scripts/train_deepUNet_2D_gen_tune.py 320 64 2 'nadam' 5  40 1  $feature_dir $output_dir $acclog_dir 1 "glorot_uniform" "binary_crossentropy" 3

# binary_crossentropy
# VarianceScaling
# lecun_normal
# he_normal
# RandomUniform 