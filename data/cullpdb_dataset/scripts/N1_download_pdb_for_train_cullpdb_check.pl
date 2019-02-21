 #! /usr/bin/perl -w
 # 
 
$num = @ARGV;
if($num != 3)
{
  die "The number of parameter is not correct!\n";
}

$pdb_list = $ARGV[0];
$outputdir = $ARGV[1];
$output_list = $ARGV[2];


$outputfasta = "$outputdir/all.fasta";
$outputresolution = "$outputdir/chain_resolution.txt";

-f "$outputfasta" || die "Failed to find $outputfasta\n";
-f "$outputresolution" || die "Failed to find $outputresolution\n"; #

## load fasta
open(IN,$outputfasta) || die "Failed to open file $outputfasta\n";
@data_content = <IN>;
close IN;
%fasta_array = ();
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
    $fasta_array{$protein}=1;
    
  }
}

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
    next;
  }
  @array = split(/\s++/,$line);
  $protein = $array[0];
  $resolution_array{$protein}=1;
}



open(LOG,">$output_list") || die "Failed to open file $output_list\n";
open(IN,$pdb_list) || die "Failed to open file $pdb_list\n";
chdir($outputdir);
$c=0;
while(<IN>)
{
  $line=$_;
  chomp $line;
  if(index($line,'IDs')>=0)
  {
    print LOG "$line\n";
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
  if(-e "$pdbname_uc-$chainid.chn" and -e "$pdbname_uc-$chainid.fasta" and exists($resolution_array{"$pdbname_uc-$chainid"}) and exists($fasta_array{"$pdbname_uc-$chainid"}))
  {
    next;
  }else{
    print "$line\n";
    print LOG "$line\n";
  }
  
  
}
close IN;
close LOG;


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
