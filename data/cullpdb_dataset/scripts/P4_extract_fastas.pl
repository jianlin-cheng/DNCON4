 #! /usr/bin/perl -w
 # 
 
$num = @ARGV;
if($num != 2)
{
  die "The number of parameter is not correct!\n";
}

$sort30_file = $ARGV[0]; # /storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/selected_sort30_db/selected_sort30_from50-500_res2.5.fasta   
$outdir = $ARGV[1]; #/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/fastas


if(!(-d $outdir))
{
  `mkdir -p $outdir`;
}

chdir($outdir);
open(IN,$sort30_file) || die "Failed to open file $sort30_file\n";
@content = <IN>;
close IN;
$protein="";
foreach(@content)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
    $protein = substr($line,1);
    $protein_full = $protein;
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
    $outfile = "$outdir/$protein.fasta";
    open(OUT,">$outfile") || die "Failed to open file $outfile\n";
    print OUT ">$protein_full\n$line\n";
    close OUT;
  }
}