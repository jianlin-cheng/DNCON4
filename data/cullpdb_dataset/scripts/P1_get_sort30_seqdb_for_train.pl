 #! /usr/bin/perl -w
 # 
 
$num = @ARGV;
if($num != 6)
{
  die "The number of parameter is not correct!\n";
}

$sort30_file = $ARGV[0];
$resolution = $ARGV[1];
$outputfile = $ARGV[2];
$resmax = $ARGV[3];
$minlen = $ARGV[4];
$maxlen = $ARGV[5];


open(IN,$resolution) || die "Failed to open file $resolution\n";
%include_pro = ();

while(<IN>)
{
  $line=$_;
  chomp $line;
  @tmp = split(/\t/,$line);

  $pro = $tmp[0];
  $res = $tmp[1];
  $type = $tmp[3];
  
  $include_pro{$pro}=$line;
  
  
}
close IN;


open(OUT,">$outputfile") || die "Failed to open file $outputfile\n";
open(IN,$sort30_file) || die "Failed to open file $sort30_file\n";
while(<IN>)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
    $protein = substr($line,1);
    next;
  }else{
    $sequence = $line;
    $len = length($sequence);
   	if(!exists($include_pro{$protein}))
  	{
  		print "Failed to find resolution for $protein\n";
  	}
    $res_info = $include_pro{$protein};
    @tmp = split(/\t/,$res_info);
    $pro = $tmp[0];
    $res = $tmp[1];
    $type = $tmp[3];
    if($res < $resmax and $type eq 'X' and $len>=$minlen and $len <=$maxlen)
    {
      print OUT ">$protein\n$sequence\n";
    }else{
      print "$protein is removed due to length: $len ".$res_info."\n";
    }

    next;
  }
}
close IN;
close OUT;
