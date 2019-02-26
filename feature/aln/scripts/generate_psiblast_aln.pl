#!/usr/bin/perl -w
# Badri Adhikari, 5-21-2017

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;
use Cwd qw();

my $fasta  = shift;
my $nr_db = shift;
my $outdir = shift;

if (not $fasta or not -f $fasta){
	print "Fasta file $fasta does not exist!\n" if ($fasta and not -f $fasta);
	print "Usage: $0 <fasta> <output-directory>\n";
	exit(1);
}

if (not $outdir){
	print 'Output directory not defined!';
	print "Usage: $0 <fasta> <output-directory>\n";
	exit(1);
}

if (not $nr_db){
        print 'Output directory not defined!';
        print "Usage: $0 <fasta> <output-directory><nr_db>\n";
        exit(1);
}

use constant{
	BLASTPATH   => '/storage/htc/bdm/tools/blast-2.2.26/bin/blastpgp',
	#REFORMAT    => '/storage/htc/bdm/tools/mview-1.64/bin/',
	REFORMAT    => abs_path(dirname($0)).'/alignhits.pl',
	CPU         => 8
};


confess 'Oops!! blast path not found!'   if not -f BLASTPATH;
confess 'Oops!! reformat file not found!'   if not -f REFORMAT;
####################################################################################################
my $id = basename($fasta, ".fasta");
system_cmd("mkdir -p $outdir") if not -d $outdir;
system_cmd("cp $fasta $outdir/") if not -f $outdir."/$id.fasta";
chdir $outdir or confess $!;

my $path = Cwd::cwd();
print "current path:$path\n";
$fasta = basename($fasta);
my $seq = seq_fasta($fasta);

####################################################################################################
print "Started [$0]: ".(localtime)."\n";

open FASTA, ">$id.jh.fasta" or confess "ERROR! Could not open $id.jh.fasta $!";
print FASTA ">$id.jh.fasta\n";
print FASTA "$seq\n";
close FASTA;

# system_cmd(BLASTPATH."  -query ".$id.".fasta -evalue .001 -inclusion_ethresh .002 -num_alignments 5000 -max_target_seqs 20000 -db ".$nr_db." -num_iterations 3 -outfmt 0 -out blast.output");
# system_cmd(REFORMAT."./mview blast.output -cycle all -out plain > blast_seq_aln.txt");
# system_cmd("awk '{print \$2}' blast_seq_aln.txt > blast.aln");
system_cmd(BLASTPATH." -i $id.jh.fasta -o blast.out -j 3 -b 75000 -e 0.001 -d ".$nr_db." > psiblast.log");
system_cmd(REFORMAT." blast.out blast.a3m");
system_cmd("egrep -v \"^>\" $id.jh.fasta  > blast.aln");
system_cmd("grep -o '\\S*\$' blast.a3m >> blast.aln");

print "\nFinished [$0]: ".(localtime)."\n";

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
