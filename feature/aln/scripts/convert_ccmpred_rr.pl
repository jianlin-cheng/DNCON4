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

system("mkdir predictions") if(!(-d "predictions"));
ccmpred2rr("$id.fasta", "jhm-1e-10.ccmpred", "./predictions/jhm-1e-10.ccmpred.rr", 6);
ccmpred2rr("$id.fasta", "jhm-1e-20.ccmpred", "./predictions/jhm-1e-20.ccmpred.rr", 6);
ccmpred2rr("$id.fasta", "jhm-1e-4.ccmpred", "./predictions/jhm-1e-4.ccmpred.rr", 6);
ccmpred2rr("$id.fasta", "jhm-e-0.ccmpred", "./predictions/jhm-e-0.ccmpred.rr", 6);
ccmpred2rr("$id.fasta", "jhm-10.ccmpred", "./predictions/jhm-10.ccmpred.rr", 6);
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
