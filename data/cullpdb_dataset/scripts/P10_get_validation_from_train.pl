#!/usr/bin/perl -w

use List::Util qw/shuffle/;
use POSIX;
use Scalar::Util qw(looks_like_number);
if (@ARGV != 3)
{
	die "need four parameters: ss_predictor, input_sequence, align_dir, output_file\n";
}

$all_list		= "$ARGV[0]";
$train_list		= "$ARGV[1]";
$val_list		= "$ARGV[2]";



open(TRAIN,">$train_list")|| die("Failed to open file $train_list \n");
open(VAL,">$val_list")|| die("Failed to open file $val_list \n");
open(IN1,"$all_list")|| die("Failed to open file $all_list \n");
@content = <IN1>;
close IN1;


@arr = shuffle @content;

$total_num = @content;
$train_num = ceil($total_num*0.9);

$num=0;
foreach $id (@arr)
{
	chomp $id;
  $num++;
  if($num<=$train_num)
  {
    print TRAIN "$id\n";
  }else{
    print VAL "$id\n";
  }
}
close IN1;
#remove the temporary files

