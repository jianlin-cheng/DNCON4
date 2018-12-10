#!/usr/bin/perl -w

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;
use Cwd qw();
use List::Util qw[min max];

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
	JACK_HH3 => abs_path(dirname($0)).'/jack_hhblits3',
	HHBLITS     => '/storage/htc/bdm/tools/hhsuite-3.0-beta.1/bin/hhblits',
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

$hhbid = "hhb-cov50";
if (-s "hhb-cov50.aln"){
	print "Looks like hhblits aln file is already here.. skipping..\n";
}else{
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


}

####################################################################################################
print "\nAlignment Summary:\n";
print 'L = '.length($seq)."\n";
system "wc -l *.aln";
print "\n";

# Apply alignment selection rule to select the best alignment file as $id.aln
#my $T = 10 * length($seq);
my $T = 128;
my $found_aln = 0;
foreach my $c (sort {$COVERAGE{$a} <=> $COVERAGE{$b}} keys %COVERAGE){
	last if $found_aln;
	my $hhbid = "hhb-cov".$c;
	confess "Oops!! Expected file $hhbid.aln not found!" if not -f "$hhbid.aln";
	if (cal_Neff("$hhbid.aln") > $T){
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
# Run Jackhmmer searching for extra seq db for hhblits

if (-s "$id.jackaln"){
	print "Looks like hhb_jhm aln file is already here.. skipping..\n";
}else{
	system_cmd(JACK_HH." $id /storage/htc/bdm/tianqi/DNCON2.5/metapsicov-2.0.3/bin $path ".$unirefdb." > $path/$id.jacklog");
}

my $naln_hhblits = count_lines("$hhbid.aln");
my $naln_jack = count_lines("$id.jackaln");
my $Nfaln_hhblits = cal_Neff("$hhbid.aln");
my $Nfaln_jack = cal_Neff("$id.jackaln");

if($Nfaln_jack > $T){
		system_cmd("cp -f $id.jackaln $id.aln");
		print "\nFinished [$0]: ".(localtime)."\n";
		exit 0;
}else{
####################################################################################################
		# Use id.a3m Run HHsearch for extra seq db on Metaclust50
		if (-s "$id.metaln"){
			print "Looks like hhb_jhm aln file is already here.. skipping..\n";
		}else{
			print "build hmmer HMM model...\n";
			system_cmd(JACKHMMER."/binaries/hmmbuild $id.hmmer $id.a3m");
			print "search HMMER hmm against database ...\n";
			my $evalue = 0.001;
			system_cmd(JACKHMMER."/binaries/hmmsearch -E 0.001  --noali --tblout  ${id}_2nd.tbl $id.hmmer $metaclust_db > $id-hmmsearch.out");

			if ((count_lines("$id-hmmsearch.out") > 75017) && (count_lines("$id.tbl")> 75003)) {
				system("head -75017 $id-hmmsearch.out > temp-hmmsearch.out");
				system_cmd("rm $id-hmmsearch.out");
				system_cmd("mv temp-hmmsearch.out $id-hmmsearch.out");
				system("head -75003 ${id}_2nd.tbl > temp.tbl");
				system_cmd("rm ${id}_2nd.tbl");
				system_cmd("mv temp.tbl ${id}_2nd.tbl");
			}
			system_cmd(JACK_HH3." $id /storage/htc/bdm/tianqi/DNCON2.5/metapsicov-2.0.3/bin $path $metaclust_db ${id}_2nd.tbl > $path/$id.jacklog");
		}
}

####################################################################################################
# Compare Neff of metaln, jackaln, hhblits and generate final aln
my $naln_metaln = count_lines("$id.metaln");
my $Nfaln_metaln = cal_Neff("$id.metaln");

if($Nfaln_metaln > $T){
		system_cmd("cp -f $id.metaln $id.aln")
}
else{
		if($Nfaln_hhblits == max($Nfaln_hhblits,$Nfaln_jack,$Nfaln_metaln)){
				system_cmd("cp -f $hhbid.aln $id.aln")
		}
		if($Nfaln_jack == max($Nfaln_hhblits,$Nfaln_jack,$Nfaln_metaln)){
				system_cmd("cp -f $id.jackaln $id.aln")
		}
		if($Nfaln_metaln == max($Nfaln_hhblits,$Nfaln_jack,$Nfaln_metaln)){
				system_cmd("cp -f $id.metaln $id.aln")
		}
}

####################################################################################################
#Filter combined aln
system_cmd("cp $id.aln ${id}_NOF.aln");
system_cmd("rm -f $id.aln");
system_cmd("perl ".FILTALN." ${id}_NOF.aln $id.aln");

####################################################################################################
# if (count_lines("$id.aln") > 75000){
	# print("More than 75,000 rows in the alignment file.. trimming..\n");
	# system_cmd("head -75000 $id.aln > temp.aln");
	# system_cmd("rm $id.aln");
	# system_cmd("mv temp.aln $id.aln");
# }

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

####################################################################################################
sub cal_Neff{
	my $file = shift;
	my $N=0;
	my $Neff=0;
	if (-s "$file.colstats"){
		print "Looks like $file alnstats file is already here.. skipping..\n";
	}else{
		system_cmd("/storage/htc/bdm/tools/MetaPSICOV/bin/alnstats $file $file.colstats $file.pairstats");
	}
	open COLSTATS, "<$file.colstats" or confess $!;
	while(<COLSTATS>){
		$N=<COLSTATS>;
		$Neff =<COLSTATS>;
		print "$file N:$N   Neff:$Neff\n";
		last;
	}
	return $Neff;
}
