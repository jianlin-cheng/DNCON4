#!/usr/bin/perl -w

use strict;

if (@ARGV ne 2 ){
  print STDERR "Usage: $0 <fasta> <file-dist>\n";
  exit;
}

my $fasta_fname = $ARGV[0];
my $dist_fname  = $ARGV[1];

my $id = $fasta_fname;
$id =~ s/\..*//;
$id =~ s/^.*\///;

open FASTA, "<" . $fasta_fname or die "Couldn't open fasta file\n";
my @lines = <FASTA>;
chomp(@lines);
close FASTA;

shift @lines;
my $seq = join('', @lines);
$seq =~ s/ //g;
my @seq = split(//, $seq);
my $seq_len = length($seq);

my %dist;
open CS, $dist_fname or die $!." $dist_fname";
while(<CS>){
	chomp $_;
	my @C = split /\s+/, $_;
	$dist{$C[0]." ".$C[1]} = $C[2];
	$dist{$C[1]." ".$C[0]} = $C[2];
}
close CS;

######### set up the distance distribution

my %distance_interval = ();
my $class_id = 0;

$distance_interval{"0|4"} = 0;

for(my $dist_start= 4; $dist_start < 24; $dist_start += 0.5)
{
  $class_id++;
  my $dist_end = $dist_start + 0.5;
  my $string = "$dist_start|$dist_end";
  $distance_interval{$string} = $class_id;
  
}
$class_id++;
$distance_interval{"24|1000000000"} = $class_id;


print "# True distance distribution map\n";
my ($atom_dist,$check,$dist_class);
for(my $i = 1; $i <= $seq_len; $i++) {
	for(my $j = 1; $j <= $seq_len; $j++) {
    if(!defined($dist{$i." ".$j}))
    {
      $atom_dist = 0;
    }else{
      $atom_dist = $dist{$i." ".$j};
    }
    $check=0;
    $dist_class=0;
    foreach my $item (keys %distance_interval)
    {
      my @tmp_array = split(/\|/,$item);
      if($atom_dist < $tmp_array[1] and $atom_dist >= $tmp_array[0])
      {
        $check=1;
        $dist_class = $distance_interval{$item};
        last;
      }
    }
    if($check ==0)
    {
      die "Warn: failed to find class for distance $atom_dist\n\n";
    }
    if($j==1)
    {
    	print "$dist_class";
    }else{
    	print " $dist_class";
    }
		
	}
	print "\n";
}
