 #! /usr/bin/perl -w
 # 
 
$num = @ARGV;
if($num != 4)
{
  die "The number of parameter is not correct!\n";
}

$pdb_list = $ARGV[0];
$outputdir = $ARGV[1];
$output_reso = $ARGV[2];
$output_fasta = $ARGV[3];

open(RES,">$output_reso") || die "Failed to open file $output_reso\n";
open(FAS,">$output_fasta") || die "Failed to open file $output_fasta\n";

$outputresolution = "$outputdir/chain_resolution.txt";

-f "$outputresolution" || die "Failed to find $outputresolution\n"; #


## load resolution
open(IN,$outputresolution) || die "Failed to open file $outputresolution\n";
@data_content = <IN>;
close IN;
%resolution_array = ();
foreach(@data_content)
{
  $line=$_;
  chomp $line;
  if(index($line,'PDBcode')>=0)
  {
    print RES "$line\n";
    next;
  }
  @array = split(/\s++/,$line);
  $protein = $array[0];
  $resolution_array{$protein}=$line;
}

open(IN,$pdb_list) || die "Failed to open file $pdb_list\n";
chdir($outputdir);
$c=0;
while(<IN>)
{
  $line=$_;
  chomp $line;
  if(index($line,'IDs')>=0)
  {
    next;
  }
  @array = split(/\s++/,$line);
  $protein = $array[0];
  if(length($protein) !=5)
  {
    print "The format of $protein is not correct!\n";
    next;
  }
  $pdbname = lc(substr($protein,0,4));
  $pdbname_uc = uc(substr($protein,0,4));
  $chainid = substr($protein,4);
  if(-e "$pdbname_uc-$chainid.chn" and -e "$pdbname_uc-$chainid.fasta" and exists($resolution_array{"$pdbname_uc-$chainid"}))
  {
    print RES $resolution_array{"$pdbname_uc-$chainid"}."\n";
    open(SEQ, "$pdbname_uc-$chainid.fasta") || die "fail to open $pdbname_uc-$chainid.fasta.\n";
    @content = <SEQ>;
    close SEQ;
    foreach $li (@content)
    {
      chomp $li;
      if(substr($li,0,1) eq '>')
      {
        print FAS uc($li)."\n";
      }else{
        print FAS "$li";
      }
    }
    print FAS "\n";
  }else{
    print "Failed to find $line\n";
  }
  
  
}
close IN;
close RES;
close FAS;


sub parse_pdb_row{
	my $row = shift;
	my $param = shift;
	my $result;
	$result = substr($row,6,5) if ($param eq "anum");
	$result = substr($row,12,4) if ($param eq "aname");
	$result = substr($row,16,1) if ($param eq "altloc");
	$result = substr($row,17,3) if ($param eq "rname");
	$result = substr($row,22,5) if ($param eq "rnum");
	$result = substr($row,21,1) if ($param eq "chain");
	$result = substr($row,30,8) if ($param eq "x");
	$result = substr($row,38,8) if ($param eq "y");
	$result = substr($row,46,8) if ($param eq "z");
	die "Invalid row[$row] or parameter[$param]" if (not defined $result);
	$result =~ s/\s+//g;
	return $result;
}
