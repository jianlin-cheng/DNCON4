#!/usr/bin/perl


if (@ARGV != 3)
{
  die "Need three arguments: dssp_dir, source_dir dest_dir\n"
}

$fasta_list = shift @ARGV; 
$summary_file = shift @ARGV;
$outputfile =  shift @ARGV;

open(OUT,">$outputfile") || die "Failed to open file $outputfile\n";
open(IN,$fasta_list) || die "Failed to open file $fasta_list\n";
@data_content = <IN>;
close IN;

%protein_selected=();
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
    $protein_selected{$protein}=1;

    next;
  }
  
}

open(IN,$summary_file) || die "Failed to open file $summary_file\n";
@data_content = <IN>;
close IN;

foreach(@data_content)
{
  $line=$_;
  chomp $line;
  if(index($line,'PDBcode')>=0)
  {
    print OUT "$line\n";
    next;
  }
  @tmp = split(/\s+/,$line);
  $pid = $tmp[0];
  if(exists($protein_selected{$pid}))
  {
    print OUT "$line\n";
  }
}

close OUT;

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
	print "Invalid row[$row] or parameter[$param]" if (not defined $result);
	$result =~ s/\s+//g;
	return $result;
}



