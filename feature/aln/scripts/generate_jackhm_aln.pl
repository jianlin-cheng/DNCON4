#!/usr/bin/perl -w
# Badri Adhikari, 9-9-2017
# Modified by Tianqi, 11-28-2017, add nr option

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;
use LWP::UserAgent;
use Time::Piece;

####################################################################################################
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
	JACKHMMER   => '/storage/htc/bdm/tools/hmmer-3.1b2-linux-intel-x86_64/binaries/jackhmmer',
	REFORMAT    => abs_path(dirname($0)).'/reformat.pl',
	#JACKHMMERDB => '/storage/htc/bdm/tools/databases/uniref/uniref90pfilt',
	#JACKHMMERDB => '/storage/htc/bdm/tools/nr_database_updated/', #modified by Tianqi
	#JACKHMMERDB => '/storage/htc/bdm/tools/uniref_updated/',
	CPU         => 4 #modified by Tianqi, default=2
};

my %EVALUE = ();
$EVALUE{'1e-20'} = 1;
$EVALUE{'1e-10'} = 2;
$EVALUE{'1e-4'}  = 3;
$EVALUE{'1'}     = 4;
$EVALUE{'10'}     = 5;

#my $JACKHMMERDB = JACKHMMERDB.$nr_db.'/'.$nr_db;
my $JACKHMMERDB = $nr_db;
print $JACKHMMERDB;
confess 'Oops!! jackhmmer not found!'   if not -f JACKHMMER;
confess 'Oops!! reformat not found!'    if not -f REFORMAT;
confess 'Oops!! jackhmmerdb not found!' if not -f $JACKHMMERDB;


####################################################################################################

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

####################################################################################################
my $id = basename($fasta, ".fasta");
system_cmd("mkdir -p $outdir") if not -d $outdir;
system_cmd("cp $fasta $outdir/") if not -f $outdir."/$id.fasta";
chdir $outdir or confess $!;
$fasta = basename($fasta);
my $seq = seq_fasta($fasta);


####################################################################################################
print "Started [$0]: ".(localtime)."\n";

# JackHmmer needs fasta sequence to be in a single line
open FASTA, ">$id.jh.fasta" or confess "ERROR! Could not open $id.jh.fasta $!";
print FASTA ">$id.jh.fasta\n";
print FASTA "$seq\n";
close FASTA;

my %jobs = ();
foreach my $e (keys %EVALUE){
	my $jhmid = "jhm-".$e;
	$jhmid = "jhm-e-0" if $e eq '1';
	next if (-e "$jhmid.aln");
	open  JOB, ">$jhmid.sh" or confess "ERROR! Could not open $jhmid.sh $!";
	print JOB "#!/bin/bash\n";
	print JOB "touch $jhmid.running\n";
	print JOB "echo \"running jackhmmer job $jhmid..\"\n";
	print JOB JACKHMMER.' --cpu '.CPU." -N 5 -E $e --incE 1e-3 -A $jhmid.ali $id.jh.fasta ".$JACKHMMERDB." > $jhmid-jackhmmer.log\n";
	print JOB "if [ ! -f \"${jhmid}.ali\" ]; then\n";
	print JOB "   mv $jhmid.running $jhmid.failed\n";
	print JOB "   echo \"jackhmmer job $jhmid failed!\"\n";
	print JOB "   exit\n";
	print JOB "fi\n";
	print JOB REFORMAT." -l 1500 -d 1500 sto a3m $jhmid.ali $jhmid.a3m\n";
	print JOB "egrep -v \"^>\" $jhmid.a3m | sed 's/[a-z]//g' > $jhmid.aln\n";
	print JOB "if [ -f \"${jhmid}.aln\" ]; then\n";
	print JOB "   rm $jhmid.running\n";
	# jackhmmer log files and .ali files use up a lot of space
	print JOB "   rm $jhmid-jackhmmer.log\n";
	print JOB "   rm $jhmid.ali\n";
	print JOB "   echo \"jackhmmer job $jhmid done.\"\n";
	print JOB "   exit\n";
	print JOB "fi\n";
	print JOB "echo \"Something went wrong! $jhmid.aln file not present!\"\n";
	print JOB "mv $jhmid.running $jhmid.failed\n";
	close JOB;
	system_cmd("chmod 755 $jhmid.sh");
	$jobs{$jhmid.".sh"} = 1;
}

foreach my $job (sort keys %jobs){
	print "Starting job $job ..\n";
	system "./$job &";
	sleep 1;
}

####################################################################################################
print("Wait until all JackHmmer jobs are done ..\n");
my $running = `find . -name "*.running" | wc -l`;
chomp $running;
confess 'Oops!! Something went wrong! No jobs are running!' if (int($running) < 0);
print "$running jobs running currently\n";
while (int($running) > 0){
	sleep 2;
	my $this_running = `find . -name "*.running" | wc -l`;
	chomp $this_running;
	$this_running = 0 if not $this_running;
	if(int($this_running) != $running){
		print "$this_running jobs running currently\n";
	}
	$running = $this_running;
}

####################################################################################################
print "\nAlignment Summary:\n";
print 'L = '.length($seq)."\n";
system "wc -l *.aln";
print "\n";

####################################################################################################
# print "Check sequences that are shorter and throw them away..\n";
# my $L = length($seq);
# open ALN, "$id.aln" or confess $!;
# open TEMP, ">temp.aln" or confess $!;
# while (<ALN>){
# 	chomp $_;
# 	if (length($_) != $L){
# 		print "Skipping - $_\n";
# 		next;
# 	}
# 	print TEMP $_."\n";
# }
# close TEMP;
# close ALN;
#
# system_cmd("mv temp.aln $id.aln");
#
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
