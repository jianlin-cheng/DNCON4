$numArgs = @ARGV;
if($numArgs != 2)
{   
	print "the number of parameters is not correct!\n";
	exit(1);
}

$infile	= "$ARGV[0]";
$outfile	= "$ARGV[1]";

open(OUT, ">$outfile") || die("Couldn't open file $outfile\n"); 
open(IN1, "$infile") || die("Couldn't open file $infile\n"); 
$c=0;
while(<IN1>)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
    $c++;
    if($c>1)
    {
      print OUT "\n";
    }
    print OUT "$line\n";
  }else{
    print OUT "$line";
  }
}
close IN1;
print OUT "\n";
close OUT;