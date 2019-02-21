#!/usr/bin/perl -w
# Tianqi Wu, 10-30-2018

#perl /storage/htc/bdm/DNCON4/feature/aln/scripts/generate_hhblits_aln.pl /storage/htc/bdm/DNCON4/feature/fasta/T0948.fasta /storage/htc/bdm/tools/databases/uniclust30_2018_08/uniclust30_2018_08  /storage/htc/bdm/DNCON4/feature/aln/hhblits/uniclust30_2018_08/T0948

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;

use constant{
	HHBLITS     => '/storage/htc/bdm/tools/hhsuite-3.0-beta.1/bin/hhblits',
	#HHBLITSDB   => '/storage/htc/bdm/tools/databases/uniclust30_2017_10/uniclust30_2017_10',
	CPU         => 2
};

my %COVERAGE = ();
$COVERAGE{50} = 1;
# $COVERAGE{60} = 1;
# $COVERAGE{70} = 2;
# $COVERAGE{80} = 3;

confess 'Oops!! hhblits not found!'     if not -f HHBLITS;
####################################################################################################
my $fasta  = shift;
my $HHBLITSDB = shift;
my $outdir = shift;

if (not $fasta or not -f $fasta){
	print "Fasta file $fasta does not exist!\n" if ($fasta and not -f $fasta);
	print "Usage: $0 <fasta> <DB> <output-directory>\n";
	exit(1);
}

if (not -f $HHBLITSDB.'_a3m_db'){
  print 'HHblits DB not found!';
	print "Usage: $0 <fasta> <DB> <output-directory>\n";
	exit(1);
};

if (not $outdir){
	print 'Output directory not defined!';
	print "Usage: $0 <fasta> <DB> <output-directory>\n";
	exit(1);
}

####################################################################################################
my $id = basename($fasta, ".fasta");
system_cmd("mkdir -p $outdir") if not -d $outdir;
system_cmd("cp $fasta $outdir/") if not -f $outdir."/$id.fasta";
chdir $outdir or confess $!;
$fasta = basename($fasta);
my $seq = seq_fasta($fasta);

# check and quit, if there are any results already
my $existing = `find . -name "*.aln" | wc -l`;
$existing = 0 if not $existing;
confess 'Oops!! There are already some alignment file in the ouput directory! Consider running in an empty directory!' if int($existing) > 0;

####################################################################################################
print "Started [$0]: ".(localtime)."\n";

my %jobs = ();
foreach my $c (keys %COVERAGE){
	my $hhbid = "hhb-cov".$c;
	open  JOB, ">$hhbid.sh" or confess "ERROR! Could not open $hhbid.sh $!";
	print JOB "#!/bin/bash\n";
	print JOB "export HHLIB=/storage/htc/bdm/tools/hhsuite-3.0-beta.1/build\n";
	print JOB "PATH=\$PATH:\$HHLIB/bin:\$HHLIB/scripts\n";
	print JOB "touch $hhbid.running\n";
	print JOB "echo \"running hhblits job $hhbid..\"\n";
	print JOB HHBLITS." -i $fasta -d ".$HHBLITSDB." -oa3m $hhbid.a3m -cpu ".CPU." -n 3 -maxfilt 500000 -diff inf -e 0.001 -id 99 -cov $c > $hhbid-hhblits.log\n";
	print JOB "if [ ! -f \"${hhbid}.a3m\" ]; then\n";
	print JOB "   mv $hhbid.running $hhbid.failed\n";
	print JOB "   echo \"hhblits job $hhbid failed!\"\n";
	print JOB "   exit\n";
	print JOB "fi\n";
	print JOB "egrep -v \"^>\" $hhbid.a3m | sed 's/[a-z]//g' > $hhbid.aln\n";
	print JOB "if [ -f \"${hhbid}.aln\" ]; then\n";
	print JOB "   mv $hhbid.running $hhbid.done\n";
	print JOB "   echo \"hhblits $hhbid job done.\"\n";
	print JOB "   exit\n";
	print JOB "fi\n";
	print JOB "echo \"Something went wrong! $hhbid.aln file not present!\"\n";
	print JOB "mv $hhbid.running $hhbid.failed\n";
	close JOB;
	system_cmd("chmod 755 $hhbid.sh");
	$jobs{$hhbid.".sh"} = 1;
}

foreach my $job (sort keys %jobs){
	print "Starting job $job ..\n";
	system "./$job &";
	sleep 1;
}

####################################################################################################
print("Wait until all HHblits jobs are done ..\n");
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
