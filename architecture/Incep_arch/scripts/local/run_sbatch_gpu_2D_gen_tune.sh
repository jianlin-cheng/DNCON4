

export HDF5_USE_FILE_LOCKING=FALSE
temp_dir=$(pwd)
gloable_dir=${temp_dir%%DNCON4*}'DNCON4'
feature_dir=$gloable_dir/data/badri_training_benchmark/feats/
output_dir=$gloable_dir/architecture/outputs/Incep_arch/new_maxout
acclog_dir=$gloable_dir/architecture/outputs/All_Validation_Acc
printf "$gloable_dir\n"

python $gloable_dir/architecture/Incep_arch/scripts/train_deepIncep_2D_gen_tune.py 150 32 6 'nadam' 3 40 1  $feature_dir $output_dir $acclog_dir 1 "glorot_uniform" "binary_crossentropy" 3

# initializer = sys.argv[12]
# loss_function = sys.argv[13]
# weight0 = int(sys.argv[14])
# weight1 = int(sys.argv[15])
# categorical_crossentropy
# binary_crossentropy
# weighted_crossentropy