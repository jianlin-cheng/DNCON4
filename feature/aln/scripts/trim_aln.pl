#!/usr/bin/perl -w
# Extract top # lines from aln
#cd /storage/htc/bdm/DNCON4/feature/aln/jackhm/uniref90_06/T0764
#perl trim_aln.pl jhm-e-0

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;
use Cwd qw();

my $id  = shift;

if (count_lines("$id.aln") > 75000){
	print("More than 75,000 rows in the alignment file.. trimming..\n");
	system_cmd("head -75000 $id.aln > temp.aln");
	system_cmd("cp $id.aln ${id}_orig.aln");
	system_cmd("mv temp.aln $id.aln");
}
####################################################################################################
sub system_cmd{
	my $command = shift;
	my $log = shift;
	confess "EXECUTE [$command]?\n" if (length($command) < 5  and $command =~ m/^rm/);
	if(defined $log){
		system("$command &> $log");
	}
	else{
		system($command);
	}
	if($? != 0){
		my $exit_code  = $? >> 8;
		confess "ERROR!! Could not execute [$command]! \nError message: [$!]";
	}
}

####################################################################################################
sub seq_fasta{
	my $file_fasta = shift;
	confess "ERROR! Fasta file $file_fasta does not exist!" if not -f $file_fasta;
	my $seq = "";
	open FASTA, $file_fasta or confess $!;
	while (<FASTA>){
		next if (substr($_,0,1) eq ">");
		chomp $_;
		$_ =~ tr/\r//d; # chomp does not remove \r
		$seq .= $_;
	}
	close FASTA;
	return $seq;
}

####################################################################################################
sub count_lines{
	my $file = shift;
	my $lines = 0;
	return 0 if not -f $file;
	open FILE, $file or confess "ERROR! Could not open $file! $!";
	while (<FILE>){
		chomp $_;
		$_ =~ tr/\r//d; # chomp does not remove \r
		next if not defined $_;
		next if length($_) < 1;
		$lines ++;
	}
	close FILE;
	return $lines;
}
