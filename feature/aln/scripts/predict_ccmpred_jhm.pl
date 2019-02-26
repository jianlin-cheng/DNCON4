#!/usr/bin/perl -w
# Badri Adhikari, 9-9-2017

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;
use LWP::UserAgent;
use Time::Piece;

####################################################################################################
my $id = shift;
my $alndir  = shift;

if (not $alndir){
	print 'ALN directory not defined!';
	exit(1);
}
chdir($alndir);
####################################################################################################
use constant{
	CCMPRED   => '/storage/htc/bdm/tools/CCMpred/bin/ccmpred',
  HOURLIMIT => 24,
	NPROC     => 8
};

confess 'Oops!! ccmpred qq not found!'.CCMPRED if not -f CCMPRED;

my %EVALUE = ();
$EVALUE{'1e-20'} = 1;
$EVALUE{'1e-10'} = 2;
$EVALUE{'1e-4'}  = 3;
$EVALUE{'1'}     = 4;
$EVALUE{'10'}     = 5;
####################################################################################################
print "Started [$0]: ".(localtime)."\n";

my %jobs = ();
foreach my $e (keys %EVALUE){
	  my $jhmid = "jhm-".$e;
	  $jhmid = "jhm-e-0" if $e eq '1';
     system_cmd("rm -f ${jhmid}_ccmpred*");
	  next if(-f "$jhmid.ccmpred");
	  open  JOB, ">${jhmid}_ccmpred.sh" or confess "ERROR! Could not open ${jhmid}_ccmpred.sh $!";
	  print JOB "#!/bin/bash\n";
	  print JOB "touch ${jhmid}_ccmpred.running\n";
	  print JOB "echo \"running ccmpred ..\"\n";
	  print JOB CCMPRED." -t ".NPROC." $jhmid.aln $jhmid.ccmpred > ccmpred.log\n";
	  print JOB "if [ -s \"$jhmid.ccmpred\" ]; then\n";
	  print JOB "   mv ${jhmid}_ccmpred.running ${jhmid}_ccmpred.done\n";
	  print JOB "   echo \"ccmpred job done.\"\n";
	  print JOB "   exit\n";
	  print JOB "fi\n";
	  print JOB "echo \"ccmpred failed!\"\n";
	  print JOB "mv ${jhmid}_ccmpred.running ${jhmid}_ccmpred.failed\n";
	  close JOB;
	system_cmd("chmod 755 ${jhmid}_ccmpred.sh");
	$jobs{"${jhmid}_ccmpred.sh"} = 1;
}

foreach my $job (sort keys %jobs){
	print "Starting job $job ..\n";
	system "./$job ";
	sleep 1;
}

####################################################################################################
print("Wait until all CCMPRED jobs are done ..\n");
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
print "\n\n";
print "Preparing predictions..\n";
system_cmd("mkdir -p predictions");
system_cmd("rm -f predictions/*");
foreach my $e (keys %EVALUE){
  my $jhmid = "jhm-".$e;
  $jhmid = "jhm-e-0" if $e eq '1';
  ccmpred2rr("$id.fasta", "$jhmid.ccmpred", "$jhmid.ccmpred.rr", 6);
  system_cmd("cp $jhmid.ccmpred.rr predictions/.");
  system_cmd("rm -f $jhmid.ccmpred*");
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
		print "[[Executing: $command]]\n";
		system($command);
	}
	if($? != 0){
		my $exit_code  = $? >> 8;
		confess "ERROR!! Could not execute [$command]! \nError message: [$!]";
	}
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
sub ccmpred2rr{
	my $file_fasta = shift;
	my $file_ccmpred = shift;
	my $file_rr = shift;
	my $seq_sep_th = shift;
	confess "ERROR! file_fasta not defined!" if !$file_fasta;
	confess "ERROR! file_ccmpred not defined!" if !$file_ccmpred;
	confess "ERROR! file_rr not defined!" if !$file_rr;
	confess "ERROR! seq_sep_th not defined!" if !$seq_sep_th;
	my %conf = ();
	open CCM, $file_ccmpred or confess $!;
	my $i = 1;
	while(<CCM>){
		my @C = split /\s+/, $_;
		for(my $j = 0; $j <= $#C; $j++){
			my $pair = $i." ".($j+1);
			$pair = ($j+1)." ".$i if ($j+1) < $i;
			my $confidence = $C[$j];
			$confidence = $conf{$pair} if (defined $conf{$pair} && $conf{$pair} > $confidence);
			$conf{$pair} = $confidence;
		}
		$i++;
	}
	close CCM;
	open RR, ">$file_rr" or confess $!;
	print RR "".seq_fasta($file_fasta)."\n";
	foreach (sort {$conf{$b} <=> $conf{$a}} keys %conf){
		my @C = split /\s+/, $_;
		next if abs($C[0] - $C[1]) < $seq_sep_th;
		print RR $_." 0 8 ".$conf{$_}."\n";
	}
	close RR;
}

####################################################################################################
sub seq_rr{
	my $file_rr = shift;
	confess ":(" if not -f $file_rr;
	my $seq;
	open RR, $file_rr or confess "ERROR! Could not open $file_rr! $!";
	while(<RR>){
		chomp $_;
		$_ =~ tr/\r//d; # chomp does not remove \r
		$_ =~ s/^\s+//;
		next if ($_ =~ /^>/);
		next if ($_ =~ /^PFRMAT/);
		next if ($_ =~ /^TARGET/);
		next if ($_ =~ /^AUTHOR/);
		next if ($_ =~ /^SCORE/);
		next if ($_ =~ /^REMARK/);
		next if ($_ =~ /^METHOD/);
		next if ($_ =~ /^MODEL/);
		next if ($_ =~ /^PARENT/);
		last if ($_ =~ /^TER/);
		last if ($_ =~ /^END/);
		# Now, I can directly merge to RR files with sequences on top
		last if ($_ =~ /^[0-9]/);
		$seq .= $_;
	}
	close RR;
	$seq =~ s/\s+//;
	confess ":( no sequence header in $file_rr" if not defined $seq;
	return $seq;
}

####################################################################################################
sub wrap_seq{
	my $seq = shift;
	confess ":(" if !$seq;
	my $seq_new = "";
	while($seq){
		if(length($seq) <= 50){
			$seq_new .= $seq;
			last;
		}
		$seq_new .= substr($seq, 0, 50)."\n";
		$seq = substr($seq, 50);
	}
	return $seq_new;
}
