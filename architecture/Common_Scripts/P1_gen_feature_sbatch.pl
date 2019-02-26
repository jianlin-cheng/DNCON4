#!/usr/bin/perl -w
use POSIX;

if (@ARGV != 7 ) {
  print "Usage: <input> <output>\n";
  exit;
}

$globaldir = $ARGV[0];
$featuredir = $ARGV[1];
$outputdir = $ARGV[2];
$acclogdir = $ARGV[3];
$sbatch_folder = $ARGV[4];
$netname = $ARGV[5];
$feature_list = $ARGV[6];

open(IN,"$feature_list") || die "Failed to open file $feature_list\n";

$c=0;
while(<IN>)
{
  
  $line = $_;
  chomp $line;
  print "Loading feature $line\n";
  $feature_indir = "$featuredir/$line";
  if(!(-d $feature_indir))
  {
	die "Failed to find direcdtory $feature_indir\n";
  }

   $run_outdir = "$outputdir/$line";
  if(!(-d $run_outdir))
  {
  	`mkdir $run_outdir`;	
  }


  $c++;
  print "\n\n###########  processing $line\n";

  $runfile="$sbatch_folder/P1_run_sbatch_$c.sh";
  print "Generating $runfile\n";
  open(SH,">$runfile") || die "Failed to write $runfile\n";
  

  print "$netname\n";

  print SH "#!/bin/bash -l\n";
  print SH "#SBATCH -J  fea_set\n";
  print SH "#SBATCH -o Fea_set-%j.out\n";
  print SH "#SBATCH --partition Lewis,hpc5\n";
  print SH "#SBATCH --nodes=1\n";
  print SH "#SBATCH --ntasks=1\n";
  print SH "#SBATCH --cpus-per-task=1\n";
  print SH "#SBATCH --mem-per-cpu=10G\n";
  print SH "#SBATCH --time 2-00:00\n";
  
  
  # #print SH "## Load Needed Modules\n";
  # print SH "module load cuda/cuda-8.0\n";
  
  # #print SH "## Force Load the Blacklisted Driver\n";
  # print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";
  
  print SH "## Activate python virtual environment\n";
  print SH "source /storage/hpc/scratch/zggc9/keras_theano/keras_virtual_env/bin/activate\n";
  print SH "module load R/R-3.3.1\n";
  print SH "export HDF5_USE_FILE_LOCKING=FALSE\n";
  print SH "GLOBAL_PATH=$globaldir\n";
  print SH "feature_dir=$feature_indir\n";
  print SH "output_dir=$run_outdir\n";
  print SH "acclog_dir=$acclogdir\n";

 
  if($netname eq "att_cov_after" ){  
    print SH "python \$GLOBAL_PATH/models/AttConv_ss/scripts/train_attcov_ss_after.py 15 10 5 nadam '6' 100 3 \$feature_dir \$outputdir\n";
  }
  elsif($netname eq "att_cov_before" ){  
    print SH "python \$GLOBAL_PATH/models/AttConv_ss/scripts/train_attcov_ss_before.py 15 10 5 nadam '6' 100 3 \$feature_dir \$outputdir\n";
  }
  elsif($netname eq "deepss_1dconv" ){  
    print SH "python \$GLOBAL_PATH/models/Deep1Dconv_ss/scripts/train_attcov_ss.py 15 10 5 nadam '6' 100 3 \$feature_dir \$outputdir\n";
  }
  elsif($netname eq "deepss_1dconv_drop" ){  
    print SH "python \$GLOBAL_PATH/models/Deep1Dconv_ss/scripts/train_deepcov_ss_drop.py 15 10  5 nadam '6' 100 3 \$feature_dir \$outputdir\n";
  }
  elsif($netname eq "deepss_1dResnet" ){
    print SH "python \$GLOBAL_PATH/models/Resnet1Dconv_ss/scripts/train_deepcovResnet_ss.py 15 10 5 nadam '6' 100 3 \$feature_dir \$outputdir\n";
  }
 elsif($netname eq "deepss_1dInception_fast" ){
    print SH "python \$GLOBAL_PATH/models/Inception1Dconv_ss/scripts/train_deepcovInception_ss_fast.py 15 28 4 nadam '6' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 28 4 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dInception' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 28 4 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dInception' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 28 4 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dInception' 'evalu' 25\n";
  }


  close SH;

}
