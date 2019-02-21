#!/usr/bin/perl -w
use POSIX;

if (@ARGV != 6 ) {
  print "Usage: <input> <output>\n";
  exit;
}

$globaldir = $ARGV[0];
$featuredir = $ARGV[1];
$outputdir = $ARGV[2];
$acclogdir = $ARGV[3];
$sbatch_folder = $ARGV[4];
$netname = $ARGV[5];

$c=0;
for($i=1;$i<=10;$i+=1)
{
  
  $c++;
  print "\n\n###########  processing filter size $i  ###########\n";

  $runfile="$sbatch_folder/P1_run_sbatch_$c.sh";
  print "Generating $runfile\n";
  open(SH,">$runfile") || die "Failed to write $runfile\n";
  

  print "$netname\n";
  $sub_netname=substr($netname,9,20);
  print SH "#!/bin/bash -l\n";
  print SH "#SBATCH -J  $sub_netname\n";
  print SH "#SBATCH -o $sub_netname-%j.out\n";
  print SH "#SBATCH --partition Lewis,hpc5\n";
  print SH "#SBATCH --nodes=1\n";
  print SH "#SBATCH --ntasks=1\n";
  print SH "#SBATCH --cpus-per-task=1\n";
  print SH "#SBATCH --mem-per-cpu=10G\n";
  print SH "#SBATCH --time 2-00:00\n";

  
  print SH "## Activate python virtual environment\n";
  print SH "source /storage/hpc/scratch/zggc9/keras_theano/keras_virtual_env/bin/activate\n";
  print SH "module load R/R-3.3.1\n";
  print SH "export HDF5_USE_FILE_LOCKING=FALSE\n";

  print SH "GLOBAL_PATH=$globaldir\n";
  print SH "feature_dir=$featuredir\n";
  print SH "output_dir=$outputdir/$i\n";
  print SH "acclog_dir=$acclogdir\n";

  # deepss_1dconv
  # deepss_1dRCNN
  # deepss_1dResnet
  # deepss_1dInception
  # deepss_1dResAtt2
  # deepss_1dCRMN
  # deepss_1dFrac
  if($netname eq "deepss_1dResnet_fast" ){  
    print SH "python \$GLOBAL_PATH/models/Resnet1Dconv_ss/scripts/train_deepcovResnet_ss_fast.py 15 37 5 nadam '5' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 37 5 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dResnet' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 37 5 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dResnet' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 37 5 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dResnet' 'evalu' 25\n";
  }
  elsif($netname eq "deepss_1dconv_fast" ){  
    print SH "python \$GLOBAL_PATH/models/Deep1Dconv_ss/scripts/train_deepcov_ss_fast.py 15 40 5 nadam '6' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 40 5 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dconv' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 40 5 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dconv' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 40 5 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dconv' 'evalu' 25\n";
  }
  elsif($netname eq "deepss_1dResAtt2_fast" ){  
    print SH "python \$GLOBAL_PATH/models/ResAtt21Dconv_ss/scripts/train_deepcovResAtt2_ss_fast.py 15 35 3 nadam '5' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 35 3 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dResAtt2' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 35 3 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dResAtt2' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 35 3 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dResAtt2' 'evalu' 25\n";
  }
  elsif($netname eq "deepss_1dCRMN_fast" ){  
    print SH "python \$GLOBAL_PATH/models/CRMN1Dconv_ss/scripts/train_deepcovCRMN_ss_fast.py 15 35 2 nadam '6' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 35 2 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dCRMN' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 35 2 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dCRMN' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 35 2 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dCRMN' 'evalu' 25\n";
  }
  elsif($netname eq "deepss_1dInception_fast" ){  
    print SH "python \$GLOBAL_PATH/models/Inception1Dconv_ss/scripts/train_deepcovInception_ss_fast.py 15 33 4 nadam '5' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 33 4 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dInception' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 33 4 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dInception' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 33 4 nadam '5' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dInception' 'evalu' 25\n";
  }
  elsif($netname eq "deepss_1dFrac_fast" ){  
    print SH "python \$GLOBAL_PATH/models/FracNet1Dconv_ss/scripts/train_deepcovFracNet_ss_fast.py 15 40 3 nadam '6' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 40 3 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dFrac' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 40 3 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dFrac' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 40 3 nadam '6' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dFrac' 'evalu' 25\n";
  }
  elsif($netname eq "deepss_1dRCNN_fast" )
  {  
    print SH "python \$GLOBAL_PATH/models/RCNN1Dconv_ss/scripts/train_deepcovRCNN_ss_fast.py 15 23 5 nadam '3' 100 3 \$feature_dir \$output_dir 25\n";
   
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst  15 23 5 nadam '3' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dRCNN' 'train' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst  15 23 5 nadam '3' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dRCNN' 'test' 25\n";
    print SH "python \$GLOBAL_PATH/lib/test_dnss.py \$GLOBAL_PATH/datasets/new20181005/casp9_10.lst  15 23 5 nadam '3' \$feature_dir \$output_dir \$acclog_dir 'deepss_1dRCNN' 'evalu' 25\n";
  }

  close SH;
  
}

