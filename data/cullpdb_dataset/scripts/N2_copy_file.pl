#!/usr/bin/perl

if ($#ARGV != 3)
{
  die "Need three arguments: dssp_dir, source_dir dest_dir\n"
}

$sort30_file = shift @ARGV; 
$feature_dir = shift @ARGV;
$output_dir = shift @ARGV;
$type =  shift @ARGV;

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

    $dssp_file= "$feature_dir/$protein.$type";
    #-f $dssp_file || die "can't read dssp file $dssp_file. \n";
    if(!(-e $dssp_file))
    {
      print "can't read file $dssp_file. \n";
      next;
    }
    `cp $feature_dir/$protein.$type $output_dir/$protein.$type`;
  }
  
}

