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
for($i=1;$i<=8;$i+=1)
{
  
  $c++;
  print "\n\n###########  processing $i  ###########\n";

  $runfile="$sbatch_folder/P1_run_sbatch_$c.sh";
  print "Generating $runfile\n";
  open(SH,">$runfile") || die "Failed to write $runfile\n";
  

  print "$netname\n";
  $sub_netname=substr($netname,9,25);
  print SH "#!/bin/bash -l\n";
  print SH "#SBATCH -J  $sub_netname\n";
  print SH "#SBATCH -o $sub_netname-%j.out\n";
  print SH "#SBATCH --partition gpu3\n";
  print SH "#SBATCH --nodes=1\n";
  print SH "#SBATCH --ntasks=1\n";
  print SH "#SBATCH --cpus-per-task=5\n";
  print SH "#SBATCH --mem-per-cpu=20G\n";
  print SH "#SBATCH --time 2-00:00\n";
  print SH "#SBATCH --qos=normal\n";
  print SH "#SBATCH --account=general-gpu\n";
  print SH "#SBATCH --gres gpu:1\n";

  
  print SH "## Activate python virtual environment\n";
  # print SH "source /storage/htc/bdm/Collaboration/Zhiye/Vir_env/DNCON4_vir/bin/activate\n";
  print SH "source /scratch/zggc9/DNCON4_vir/bin/activate\n";
  print SH "module load cuda/cuda-9.0.176\n";
  print SH "module load cudnn/cudnn-7.1.4-cuda-9.0.176\n";
  print SH "module load R/R-3.3.1\n";
  print SH "export HDF5_USE_FILE_LOCKING=FALSE\n";
  print SH "export GPUARRAY_FORCE_CUDA_DRIVER_LOAD=\"\"\n";

  print SH "gloable_dir=$globaldir\n";
  print SH "feature_dir=$featuredir\n";
  print SH "output_dir=$outputdir\n";
  print SH "acclog_dir=$acclogdir\n";

  # @filters=(16,20,24,28,32,36,40,48,56,64);
  # $seed=int(rand(10));
  # $filternum=$filters[$seed];
  # $layernum=int(rand(9))+2; 
  # $kernelsize=int(rand(2))+3; 

  $filternum=28;
  $layernum=6; 
  $kernelsize=3; 
  # @batchvec=(2,4,8,16,24,28,32,36);
  # "glorot_uniform" "weighted_crossentropy" 15 1
  # @initializers=('\'RandomUniform\'','\'lecun_uniform\'','\'VarianceScaling\'');
  # $seed=int(rand(3));
  $initializer='\'VarianceScaling\'';
  # @loss_functions=("weighted_crossentropy", "categorical_crossentropy", "binary_crossentropy");
  # $seed=int(rand(3));
  # $loss_function=$loss_functions[$seed];
  $loss_function = '\'weighted_crossentropy\'';
  @weights=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
  # $w_seed=int(rand(13));
  $weight=$weights[$i];
  $batchsize = 1;
  print "filternum $filternum; layernum $layernum; kernelsize $kernelsize; batchsize $batchsize; initializer $initializer; loss_function $loss_function; weight $weight\n";
  #INCEP
  # $filternum=int(32);
  # $layernum=int(5);
  # $kernelsize=int(4);
  #RES
  # $filternum=int(32);
  # $layernum=int(4);
  # $kernelsize=int(3);
  # $batchsize=$i;
  #RCNN 

  # ###All NetName
  # DNCON4_1d2dCNN
  # DNCON4_1d2dCRMN
  # DNCON4_1d2dFRAC
  # DNCON4_1d2dINCEP
  # DNCON4_1d2dRCNN
  # DNCON4_1d2dRES
  # DNCON4_1d2dRESATT
  # DNCON4_1d2dRCINCEP
  if($netname eq "DNCON4_1d2dCNN" ){  
    print SH "python \$gloable_dir/architecture/CNN_arch/scripts/train_deepCNN.py 150 $filternum  $layernum 'nadam' $kernelsize  50 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_1d2dRCNN" ){  
    print SH "python \$gloable_dir/architecture/RCNN_arch/scripts/train_deepRCNN.py 150 $filternum  $layernum 'nadam' $kernelsize  50 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_1d2dINCEP" ){  
    print SH "python \$gloable_dir/architecture/Incep_arch/scripts/train_deepIncep.py 150 $filternum  $layernum 'nadam' $kernelsize  50 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_1d2dRES" ){  
    print SH "python \$gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet.py 150 $filternum  $layernum 'nadam' $kernelsize  30 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_2dCNN" ){  
    print SH "python \$gloable_dir/architecture/CNN_arch/scripts/train_deepCNN_2D_gen.py 150 $filternum  $layernum 'nadam' $kernelsize  50 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_2dRCNN" ){  
    print SH "python \$gloable_dir/architecture/RCNN_arch/scripts/train_deepRCNN_2D_gen.py 150 $filternum  $layernum 'nadam' $kernelsize  50 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_2dINCEP" ){  
    print SH "python \$gloable_dir/architecture/Incep_arch/scripts/train_deepIncep_2D_gen.py 150 $filternum  $layernum 'nadam' $kernelsize  50 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_2dRES" ){  
    print SH "python \$gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen.py 150 $filternum  $layernum 'nadam' $kernelsize  50 1 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }
  elsif($netname eq "DNCON4_2dINCEP_Z" ){  
    print SH "python \$gloable_dir/architecture/Incep_arch/scripts/train_deepIncep_2D_gen_tune.py 150 $filternum  $layernum 'nadam' $kernelsize  30 5 \$feature_dir \$output_dir \$acclog_dir $batchsize $initializer $loss_function $weight\n";
  }
  elsif($netname eq "DNCON4_2dRES_Z" ){  
    print SH "python \$gloable_dir/architecture/ResNet_arch/scripts/train_deepResNet_2D_gen_tune.py 150 $filternum  $layernum 'nadam' $kernelsize  30 2 \$feature_dir \$output_dir \$acclog_dir $batchsize $initializer $loss_function $weight\n";
  }
  elsif($netname eq "DNCON4_1d2dFRAC" ){  
    print SH "python \$gloable_dir/architecture/FracNet_arch/scripts/train_deepFracNet.py 150 $filternum  $layernum 'nadam' $kernelsize  30 5 \$feature_dir \$output_dir \$acclog_dir $batchsize\n";
  }


  close SH;

}

