#!/usr/bin/perl -w

##########################################################################
# Extract fasta from PDB model with specified input
#
# 2 arguments
# Output:   fasta file from pdb model
# run: perl pdb2fasta.pl <pdb file> <pdb name> <fasta outfile>
# Example :  perl pdb2fasta.pl domain0.atom domain0 domain0.fasta  
#                         
# Author: Jie Hou
# Date: 05/11/2016
##########################################################################

use Carp;
our %AA3TO1 = qw(ALA A ASN N CYS C GLN Q HIS H LEU L MET M PRO P THR T TYR Y ARG R ASP D GLU E GLY G ILE I LYS K PHE F SER S TRP W VAL V);
our %AA1TO3 = reverse %AA3TO1;

$num = @ARGV;
if($num != 4)
{
	die  "The number of parameter is not correct!\n";
}


$fasta_file = shift @ARGV; 
$feature_dir = shift @ARGV;
$output_dir = shift @ARGV;
$type =  shift @ARGV;

open(IN,$fasta_file) || die "Failed to open file $fasta_file\n";
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
  
    $file_PDB= "$feature_dir/$protein.$type";
    #-f $dssp_file || die "can't read dssp file $dssp_file. \n";
    if(!(-e $file_PDB))
    {
      print "can't read file $dssp_file. \n";
      next;
    }
    $fasta_out= "$output_dir/$protein.fasta";
    print "Processing $protein\n";
    $domain_name = $protein;
  
    open SEQUENCE, ">$fasta_out" or die "Failed to open $fasta_out\n";
    
    my $seq = "";
    open(INPUTPDB, "$file_PDB") || die "ERROR! Could not open $file_PDB\n";
    while(<INPUTPDB>){
    	next if $_ !~ m/^ATOM/;
    	next unless (parse_pdb_row($_,"aname") eq "CA");
    	confess "ERROR!: ".parse_pdb_row($_,"rname")." residue not defined! \nFile: $file_PDB! \nLine : $_" if (not defined $AA3TO1{parse_pdb_row($_,"rname")});
    	my $res = $AA3TO1{parse_pdb_row($_,"rname")};
    	$seq .= $res;
    }
    close INPUTPDB;
    if (length($seq) < 1){
    	print "WARNING! $file_PDB has less than 1 residue ($seq)!\n";
    }
    print SEQUENCE ">$domain_name\n$seq\n";
    
    close SEQUENCE;
  }
}
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