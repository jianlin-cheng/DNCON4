 #! /usr/bin/perl -w
 # 
 
$num = @ARGV;
if($num != 3)
{
  die "The number of parameter is not correct!\n";
}

$sort30_file = $ARGV[0];
$pdb_dir = $ARGV[1];
$outdir = $ARGV[2];

chdir($outdir);
open(IN,$sort30_file) || die "Failed to open file $sort30_file\n";
while(<IN>)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
    $protein = substr($line,1);
    $atomfile = "$pdb_dir/$protein.atom.gz";
    if(!(-e $atomfile))
    {
      print "!!!!!!!!!! warning: failed to find $atomfile\n";
      next;
    }else{
      `cp $atomfile $outdir`;
      `gunzip $protein.atom.gz`;
    }
  }
}
close IN;