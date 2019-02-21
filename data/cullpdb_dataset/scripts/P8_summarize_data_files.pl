#!/usr/bin/perl

#######pdb2dssp.pl############################## 
#convert pdb format file to dssp format file.
#usage: pdb2dssp.pl source_dir dest_dir
#the dssp file will have the same file name with pdb file except with differnt
#suffix ".dssp.gz". The dssp file is compressed by gzip. 
#Assumption: source file name format: *.Z and compress by gzip. 
#output: pdb_prefix.dssp.gz 
#Author: Jianlin Cheng, 5/28/2003
###############################################
use Carp;
our %AA3TO1 = qw(ALA A ASN N CYS C GLN Q HIS H LEU L MET M PRO P THR T TYR Y ARG R ASP D GLU E GLY G ILE I LYS K PHE F SER S TRP W VAL V);
our %AA1TO3 = reverse %AA3TO1;


if (@ARGV != 3)
{
  die "Need three arguments: dssp_dir, source_dir dest_dir\n"
}

$fasta_list = shift @ARGV; 
$source_dir = shift @ARGV;
$out_dir = shift @ARGV;

-d "$out_dir/dssp"  || die "Failed to find $out_dir/dssp\n";
-d "$out_dir/fasta"  || die "Failed to find $out_dir/fasta\n";
-d "$out_dir/chains"  || die "Failed to find $out_dir/chains\n";


open(IN,$fasta_list) || die "Failed to open file $fasta_list\n";
@data_content = <IN>;
close IN;

$pro_num=0;
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

    $dssp_file= "$source_dir/dssp/$protein.dssp";
    $pdb_file= "$source_dir/chains/$protein.chn";
    $fasta_file= "$source_dir/fasta/$protein.fasta";
    #-f $dssp_file || die "can't read dssp file $dssp_file. \n";
    if(!(-e $dssp_file) or !(-e $pdb_file) or !(-e $fasta_file))
    {
      print "can't read dssp file $dssp_file or $pdb_file or $fasta_file. \n";
      next;
    }
    
    $pos = rindex($dssp_file, "/");
    if ($pos >= 0)
    {
    	$tmp_file = substr($dssp_file, $pos + 1) . ".tmp";
    }
    else
    {
    	$tmp_file = $dssp_file . ".tmp";
    }
    
    `cp $dssp_file $out_dir/dssp/`;
    `cp $pdb_file $out_dir/chains/`;
    `cp $fasta_file $out_dir/fasta/`;
    
    
    
  }
  
}



