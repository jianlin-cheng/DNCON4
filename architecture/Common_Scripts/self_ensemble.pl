#!/usr/bin/perl -w

# perl /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/paper_dataset/State1_Nets_On_OldFea/emsemble/ensemble_methods.pl /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/ /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/features_win1_with_atch/ /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/paper_dataset/State1_Nets_On_OldFea/emsemble/ /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/paper_dataset/State1_Nets_On_OldFea/outputs/ensemble_out  /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/paper_dataset/State1_Nets_On_OldFea/emsemble/model_test.list



use POSIX;

if (@ARGV != 5) {
  print "Usage: <input> <output>\n";
  exit;
}

$globaldir = $ARGV[0]; #/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/
$featuredir = $ARGV[1]; #/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/features_win1_with_atch/
$modeldir = $ARGV[2]; #/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/models/*/self_ensemble_results
$outputdir = $ARGV[3];#/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/output/ensemble_out
$netname = $ARGV[4];#deepss_1dconv deepss_1dCRMN deepss_1dFrac deepss_1dInception deepss_1dRCNN deepss_1dResAtt2 deepss_1dResnet


for($i=1;$i<=10;$i+=1)
{
  $netname=$_;
  chomp $netname;
  print "\n\n###########  processing network $netname  ###########\n";

  $modelfile = "$modeldir/$i/model-train-$netname.json";
  $weightfile= "$modeldir/$i/model-train-weight-$netname-best-val.h5";
  if(!(-e $modelfile))
  {
    die "Failed to find model file $modelfile\n";
  }
  if(!(-e $weightfile))
  {
    die "Failed to find model file $weightfile\n";
  }

  $GLOBAL_PATH=$globaldir;
  $output_dir="$outputdir/$netname/$i";
  if(!(-d $output_dir))
  {
    `mkdir $output_dir`;
  }
  print("python $GLOBAL_PATH/lib/predict_dnss2.py $GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst $modelfile $weightfile  $output_dir $featuredir $netname train\n");
  system("python $GLOBAL_PATH/lib/predict_dnss2.py $GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst $modelfile $weightfile  $output_dir $featuredir $netname train");
  system("python $GLOBAL_PATH/lib/predict_dnss2.py $GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst $modelfile $weightfile  $output_dir $featuredir $netname test");
  system("python $GLOBAL_PATH/lib/predict_dnss2.py $GLOBAL_PATH/datasets/new20181005/casp9_10.lst $modelfile $weightfile  $output_dir $featuredir $netname evalu");
}
close IN;


##### average train data
#python /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2//lib/average_ss_prob_dnss2.py /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2//datasets/new20181005/adj_dncon-train.lst /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/paper_dataset/State1_Nets_On_OldFea/emsemble/model_test.list  /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/paper_dataset/State1_Nets_On_OldFea/outputs/ensemble_out/old/  /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/paper_dataset/State1_Nets_On_OldFea/outputs/ensemble_out/old/avg_out deepss_1dconv_1dCRMN  train

$avg_out_dir="$outputdir/$netname";
if(!(-d $avg_out_dir))
{
  `mkdir $avg_out_dir`;
}
print("python /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/lib/average_ss_prob_dnss2.py $GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst $netfile  $outputdir/  $avg_out_dir $netname  train\n");
system("python /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/lib/average_ss_prob_dnss2.py $GLOBAL_PATH/datasets/new20181005/adj_dncon-train.lst $netfile  $outputdir/  $avg_out_dir $netname  train");

##### average validation data
print("python /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/lib/average_ss_prob_dnss2.py $GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst $netfile  $outputdir/  $avg_out_dir $netname  test\n");
system("python /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/lib/average_ss_prob_dnss2.py $GLOBAL_PATH/datasets/new20181005/adj_dncon-test.lst $netfile  $outputdir/  $avg_out_dir $netname  test");

##### average validation data
print("python /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/lib/average_ss_prob_dnss2.py $GLOBAL_PATH/datasets/new20181005/casp9_10.lst $netfile  $outputdir/  $avg_out_dir $netname  evalu\n");
system("python /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/lib/average_ss_prob_dnss2.py $GLOBAL_PATH/datasets/new20181005/casp9_10.lst $netfile  $outputdir/  $avg_out_dir $netname  evalu");

# test_list=(sys.argv[1]) 
# model_list=(sys.argv[2]) 
# network_pred_dir=sys.argv[3]
# CV_dir=sys.argv[4]
# model_prefix=(sys.argv[5])
# tag=(sys.argv[6])