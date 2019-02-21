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


%maxsa = (A=>106, B=>160, C=>135, D=>163, E=>194, F=>197,  G=>84, H=>184, I=>169, K=>205, L=>164, M=>188,
		 N=>157, P=>136, Q=>198, R=>248, S=>130, T=>142, V=>142, W=>227, X=>180, Y=>222, Z=>196); 
      
if ($#ARGV != 2)
{
  die "Need three arguments: dssp_dir, source_dir dest_dir\n"
}

$sort30_file = shift @ARGV; 
$dssp_dir = shift @ARGV;
$dssp2dataset_script =  shift @ARGV;

open(IN,$sort30_file) || die "Failed to open file $sort30_file\n";
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

    $dssp_file= "$dssp_dir/$protein.dssp";
    #-f $dssp_file || die "can't read dssp file $dssp_file. \n";
    if(!(-e $dssp_file))
    {
      print "can't read dssp file $dssp_file. \n";
      next;
    }
    $ssa_set= "$dssp_dir/$protein.ssa";
    $dssp_set= "$dssp_dir/$protein.set";
    $pos = rindex($dssp_file, "/");
    if ($pos >= 0)
    {
    	$tmp_file = substr($dssp_file, $pos + 1) . ".tmp";
    }
    else
    {
    	$tmp_file = $dssp_file . ".tmp";
    }
    
  
    
    #dssp to dataset
    system("$dssp2dataset_script $dssp_file $tmp_file");
    open(RES, $tmp_file) || die "can't read dssp 2 data set output.\n";
    @content = <RES>;
    close RES;
    `rm $tmp_file`; 
    
    open(SSA, ">$ssa_set") || die "can't create dssp dataset file.\n";
    open(SET, ">$dssp_set") || die "can't create dssp dataset file.\n";
    print SSA "#\tAA\tStruct\tRSA\n";
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
     
      $seq = join("", @vec_seq);
      $seq_ss = join("", @vec_ss);
      chomp $seq;
      #if($line ne $seq) # dssp seq doesn't need be same as original seq
      #{
      #  print "The fasta sequence not match dssp seq in $dssp_file\n$line\n$seq\n\n";
      #  next;
      #}
      for($indx=1;$indx<=@vec_seq;$indx++)
      {
         $rsa = sprintf("%.5f",$vec_sa[$indx-1]/100);
         print SSA "$indx\t".$vec_seq[$indx-1]."\t".$vec_ss[$indx-1]."\t".$rsa."\n"; 
      }
      print SET ">$protein\n$seq\n$seq_ss\n";
    	#print SET "$name $resolution $length\n$seq\n$seq_ss\n$seq_sa";
     
    }
    close SET;
  }
  
}

