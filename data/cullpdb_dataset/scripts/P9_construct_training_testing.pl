 #! /usr/bin/perl -w
 # 
 
$num = @ARGV;
if($num != 4)
{
  die "The number of parameter is not correct!\n";
}

$reso_list = $ARGV[0];
$fasta_list = $ARGV[1];
$train_output = $ARGV[2];
$test_output = $ARGV[3];
$min_length = 30;

open(TRAIN,">$train_output") || die "Failed to open file $train_output\n";
open(TRAININFO,">$train_output.info") || die "Failed to open file $train_output\n";
open(TEST,">$test_output") || die "Failed to open file $test_output\n";
open(TESTINFO,">$test_output.info") || die "Failed to open file $test_output\n";


-f "$reso_list" || die "Failed to find $reso_list\n"; #
-f "$fasta_list" || die "Failed to find $fasta_list\n"; #

%test_year =();
$test_year{17}=1;
$test_year{18}=1;

%selected_mon =();
$selected_mon{'JAN'}=1;
$selected_mon{'FEB'}=1;
$selected_mon{'MAR'}=1;
$selected_mon{'APR'}=1;


## load resolution
open(IN,$reso_list) || die "Failed to open file $reso_list\n";
@data_content = <IN>;
close IN;
%resolution_array = ();
foreach(@data_content)
{
  $line=$_;
  chomp $line;
  if(index($line,'PDBcode')>=0)
  {
    next;
  }
  @array = split(/\s++/,$line);
  $protein = $array[0];
  $resolution_array{$protein}=$line;
}

open(IN,$fasta_list) || die "Failed to open file $fasta_list\n";
@data_content = <IN>;
close IN;
foreach(@data_content)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
    $protein = substr($line,1);
    
    if(index($protein,'|')>0) #1A8L:A|PDBID|CHAIN|SEQUENCE
    {
      @tmp = split(/\|/,$protein);
      $protein = $tmp[0];
    }
  
    if(index($protein,' ')>0) #1A8L:A|PDBID|CHAIN|SEQUENCE
    {
      @tmp = split(/\s/,$protein);
      $protein = $tmp[0];
    }
    $protein =~ s/\:/\-/g;

    next;
  }else{
    $len = length($line);
    if($len < $min_length)
    {
      next;
    }
    $info = $resolution_array{$protein};
    @tmp = split(/\s++/,$info);
    $date = pop @tmp;
    chomp $date;
    @tmp2 = split('-',$date);
    $year = pop @tmp2;
    $mon = pop @tmp2;
    if($year == 18 or $year == 17)
    {
      print TEST "$protein\n";
      print TESTINFO "$info $len\n";
    }elsif($year ==16 and !exists($selected_mon{$mon}))
    {
      print TEST "$protein\n";
      print TESTINFO "$info $len\n";
    }else
    {
      print TRAIN "$protein\n";
      print TRAININFO "$info $len\n";
    }
    
  }
}
