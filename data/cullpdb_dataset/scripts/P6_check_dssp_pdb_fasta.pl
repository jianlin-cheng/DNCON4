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


if (@ARGV != 6)
{
  die "Need three arguments: dssp_dir, source_dir dest_dir\n"
}

$fasta_list = shift @ARGV; 
$pdb_dir = shift @ARGV;
$dssp_dir = shift @ARGV;
$fasta_dir = shift @ARGV;
$dssp2dataset_script =  shift @ARGV;
$outputfile =  shift @ARGV;

open(OUT,">$outputfile") || die "Failed to open file $outputfile\n";
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

    $dssp_file= "$dssp_dir/$protein.dssp";
    $pdb_file= "$pdb_dir/$protein.chn";
    $fasta_file= "$fasta_dir/$protein.fasta";
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
    
    ### get pdb sequence
    $pdb_seq = "";
    open(INPUTPDB, "$pdb_file") || die "ERROR! Could not open $pdb_file\n";
    while(<INPUTPDB>){
    	next if $_ !~ m/^ATOM/;
    	next unless (parse_pdb_row($_,"aname") eq "CA");
    	confess "ERROR!: ".parse_pdb_row($_,"rname")." residue not defined! \nFile: $file_PDB! \nLine : $_" if (not defined $AA3TO1{parse_pdb_row($_,"rname")});
    	my $res = $AA3TO1{parse_pdb_row($_,"rname")};
    	$pdb_seq .= $res;
    }
    close INPUTPDB;
    if (length($pdb_seq) < 1){
    	print "WARNING! $file_PDB has less than 1 residue ($seq)!\n";
      sleep(5);
      next;
    }   
    
    
    ### get fasta sequence
    open(RES, $fasta_file) || die "can't read $fasta_file.\n";
    @content = <RES>;
    close RES; 
    shift @content;
    $fasta_seq = shift @content;
    chomp $fasta_seq;
    
    
    #dssp seq
    system("$dssp2dataset_script $dssp_file $tmp_file");
    open(RES, $tmp_file) || die "can't read dssp 2 data set output.\n";
    @content = <RES>;
    close RES;
    `rm $tmp_file`; 
    
    while (@content)
    {
    	$name = shift @content;
    	$length = shift @content;
    	$seq = shift @content;
    	$mapping = shift @content;
    	$ss = shift @content;
    	$bp1 = shift @content;
    	$bp2 = shift @content;
    	$sa = shift @content;
    	$xyz = shift @content;
    	$blank = shift @content;
      
      chomp $seq;
      chomp $ss;
      chomp $sa;
      $ss =~ s/\./C/g;
      $ss =~ s/[GI]/H/g;
      $ss =~ s/B/E/g; 
      $ss =~ s/[TS]/C/g;
    	#check integrity before proceed
    	@vec_seq = split(/\s+/, $seq);
    	@vec_ss = split(/\s+/, $ss);
    	@vec_sa = split(/\s+/, $sa);
    	if ($length != @vec_seq || $length != @vec_ss || $length != @vec_sa)
    	{
    		die "$name, in generated set from dssp file, length is not consistent.\n";
    		next;
    	} 
     
      $dssp_seq = join("", @vec_seq);
      chomp $dssp_seq;
      if($fasta_seq ne $dssp_seq or $fasta_seq ne $pdb_seq) # dssp seq doesn't need be same as original seq
      {
        print "The fasta sequence not match dssp and fasta seq in $dssp_file or $pdb_file or $fasta_file\n$fasta_seq\n$dssp_seq\n$pdb_seq\n\n";
        next;
      }
      
      if(length($fasta_seq)>700 or length($fasta_seq)<26)
      {
        next;
      }
      $pro_num++;
      print "$protein\n";
      print OUT ">$protein\n$fasta_seq\n";
          
    }
    close SET;
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



