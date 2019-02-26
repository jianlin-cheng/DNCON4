#!/usr/bin/perl -w
# Badri Adhikari, 5-21-2017

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;
use Cwd qw();

my $fasta  = shift;
my $outdir = shift;
my $unirefdb = "/storage/htc/bdm/tools/uniref_updated/uniref90_10/uniref90";
my $uniprotdb = "/storage/htc/bdm/tools/databases/uniclust30_2018_08/uniclust30_2018_08";
my $metaclust_db = "/storage/htc/bdm/tools/databases/metaclust_50/metaclust_50";

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

use constant{
	JACKHMMER   => '/storage/htc/bdm/tools/hmmer-3.1b2-linux-intel-x86_64',
	REFORMAT    => abs_path(dirname($0)).'/reformat.pl',
	FILTALN  => abs_path(dirname($0)).'/filter_aln.pl',
	#JACKHMMERDB => '/storage/htc/bdm/tools/databases/uniref/uniref90pfilt',
	#JACK_HHDB => '/storage/htc/bdm/tools/databases/uniref/uniref90.fasta',
	JACK_HH => abs_path(dirname($0)).'/jack_hhblits',
	HHBLITS     => '/storage/htc/bdm/tools/hhsuite-3.0-beta.1/bin/hhblits',
	FASINLINE => abs_path(dirname($0)).'/convert_fasta_inline.pl',
	PARSE2HMM => abs_path(dirname($0)).'/parse_hmmer3_fast_V5.pl',
  #HHBLITS => '/storage/htc/bdm/tools/HHsuite/hhsuite-2.0.16/bin/hhblits',
	#HHBLITSDB   => '/storage/htc/bdm/tools/databases/uniprot20_2016_02/uniprot20_2016_02',
	CPU         => 8
};

my %COVERAGE = ();
$COVERAGE{50} = 1;

confess 'Oops!! jackhmmer not found!'   if not -d JACKHMMER;
confess 'Oops!! reformat not found!'    if not -f REFORMAT;
#confess 'Oops!! combinealn not found!'    if not -f COMBALN;
confess 'Oops!! jackhmmerdb not found!' if not -f $unirefdb;
confess 'Oops!! hhblits not found!'     if not -f HHBLITS;
confess 'Oops!! hhblitsdb not found!'   if not -f $uniprotdb.'_a3m_db';

####################################################################################################
my $id = basename($fasta, ".fasta");
system_cmd("mkdir -p $outdir") if not -d $outdir;
system_cmd("cp $fasta $outdir/") if not -f $outdir."/$id.fasta";
chdir $outdir or confess $!;

my $path = Cwd::cwd();
print "current path:$path\n";
$fasta = basename($fasta);
my $seq = seq_fasta($fasta);

# check and quit, if there are any results already
my $existing = `find . -name "*.aln" | wc -l`;
$existing = 0 if not $existing;
#confess 'Oops!! There are already some alignment file in the ouput directory! Consider running in an empty directory!' if int($existing) > 0;

####################################################################################################
print "Started [$0]: ".(localtime)."\n";

my ($jhmid,$hhbid);
my %jobs = ();

foreach my $c (keys %COVERAGE){
	$hhbid = "hhb-cov".$c;
	open  JOB, ">$hhbid.sh" or confess "ERROR! Could not open $hhbid.sh $!";
	print JOB "#!/bin/bash\n";
	print JOB " export HHLIB=/storage/htc/bdm/tools/hhsuite-3.0-beta.1/build\n";
  print JOB "PATH=\$PATH:\$HHLIB/bin:\$HHLIB/scripts\n";
	print JOB "touch $hhbid.running\n";
	print JOB "echo \"running hhblits job $hhbid..\"\n";
	print JOB HHBLITS." -i $fasta -d ".$uniprotdb." -oa3m $id.a3m -cpu ".CPU." -n 3 -diff inf -e 0.001 -id 99 -cov $c > $hhbid-hhblits.log\n";
	print JOB "if [ ! -f \"${id}.a3m\" ]; then\n";
	print JOB "   mv $hhbid.running $hhbid.failed\n";
	print JOB "   echo \"hhblits job $hhbid failed!\"\n";
	print JOB "   exit\n";
	print JOB "fi\n";
	print JOB "egrep -v \"^>\" $id.a3m | sed 's/[a-z]//g' > $hhbid.aln\n";
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
my $running_task =`find . -name "*.running"`;
my $running = `find . -name "*.running" | wc -l`;
chomp $running;
confess 'Oops!! Something went wrong! No jobs are running!' if (int($running) < 0);
print "$running_task $running jobs running currently\n";
while (int($running) > 0){
	sleep 2;
	$running_task =`find . -name "*.running"`;
	#print "$running_task currently";
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

# Apply alignment selection rule to select the best alignment file as $id.aln
# Increasing this threshold to 5 (from 2.5) after observing the case of T0855 where e-10 has 321 rows and e-4 has 11K rows
#my $T = 10 * length($seq);
my $T = 3000;
my $found_aln = 0;
foreach my $c (sort {$COVERAGE{$a} <=> $COVERAGE{$b}} keys %COVERAGE){
	last if $found_aln;
	my $hhbid = "hhb-cov".$c;
	confess "Oops!! Expected file $hhbid.aln not found!" if not -f "$hhbid.aln";
	if (count_lines("$hhbid.aln") > $T){
		print("Copying $hhbid.aln as $id.aln\n");
		system_cmd("echo \"cp $hhbid.aln $id.aln\" > result.txt");
		system_cmd("cp $hhbid.aln $id.aln");
		$found_aln = 1;
		last;
	}
}

if($found_aln){
	print "HHblits jobs have enough alignments! Not running JackHmmer!\n";
	print "\nFinished [$0]: ".(localtime)."\n";
	exit 0;
}

####################################################################################################
# Run Jackhmmer serching for extre seq db for hhblits

system_cmd(JACK_HH." $id /storage/htc/bdm/tianqi/DNCON2.5/metapsicov-2.0.3/bin $path ".$unirefdb." > $path/$id.jacklog");

my $naln_hhblits = count_lines("$hhbid.aln");
my $naln_jack = count_lines("$id.jackaln");

if ($naln_jack > $naln_hhblits)
{
  system_cmd("cp -f $id.jackaln $id.aln")
}
else
{
  system_cmd("cp -f $hhbid.aln $id.aln");
}

####################################################################################################
# Use id.a3m Run HHsearch for extre seq db on Metaclust50
print "build hmmer HMM model...\n";
system_cmd(JACKHMMER."/binaries/hmmbuild $id.hmmer $id.a3m");
print "search HMMER hmm against database ...\n";
my $evalue = 0.001;
system_cmd(JACKHMMER."/binaries/hmmsearch -E 0.001  --noali --tblout  $id.tbl $id.hmmer $metaclust_db > $id-hmmsearch.out");

if ((count_lines("$id-hmmsearch.out") > 75017) && (count_lines("$id.tbl")> 75003)) {
	system("head -75017 $id-hmmsearch.out > temp-hmmsearch.out");
	system_cmd("rm $id-hmmsearch.out");
	system_cmd("mv temp-hmmsearch.out $id-hmmsearch.out");
	system("head -75003 $id.tbl > temp.tbl");
	system_cmd("rm $id.tbl");
	system_cmd("mv temp.tbl $id.tbl");
}

system_cmd(JACKHMMER."/easel/miniapps/esl-sfetch -f $metaclust_db $id.tbl > $id.fseqs");
system_cmd("perl ".FASINLINE." $id.fseqs $id.fseqs.inline");
system_cmd("perl ".PARSE2HMM." $id-hmmsearch.out $fasta $id.fseqs.inline $outdir $evalue > PARSE2HMM.log");

system_cmd("cp $id.hmmer.msa  $id.metaln");
system_cmd("cat $id.metaln >> $id.aln");

####################################################################################################
#Filter combined aln
system_cmd("cp $id.aln ${id}_NOF.aln");
system_cmd("rm -f $id.aln");
system_cmd("perl ".FILTALN." ${id}_NOF.aln $id.aln");

####################################################################################################
if (count_lines("$id.aln") > 120000){
	print("More than 120,000 rows in the alignment file.. trimming..\n");
	system_cmd("head -120000 $id.aln > temp.aln");
	system_cmd("rm $id.aln");
	system_cmd("mv temp.aln $id.aln");
}

####################################################################################################
print "Check sequences that are shorter and throw them away..\n";
my $L = length($seq);
open ALN, "$id.aln" or confess $!;
open TEMP, ">temp.aln" or confess $!;
while (<ALN>){
	chomp $_;
	if (length($_) != $L){
		print "Skipping - $_\n";
		next;
	}
	print TEMP $_."\n";
}
close TEMP;
close ALN;

system_cmd("mv temp.aln $id.aln");


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
