#!/usr/bin/perl -w
#perl run_jackhm_sbatch.pl /storage/htc/bdm/DNCON4/feature/fasta /storage/htc/bdm/DNCON4/feature/aln/jackhm/uniref50_06 /storage/htc/bdm/DNCON4/feature/aln/jackhm/uniref50_06_sbatch uniref50_06/uniref50

use POSIX;

if (@ARGV != 4 ) {
  print "Usage: <fasta> <outdir> <sbatch_dir> <nr_db>\n";
  exit;
}


$fastadir = $ARGV[0]; #/storage/htc/bdm/tools/DNCON2/fastas/
$outputdir = $ARGV[1]; #.../pssm/nr90/
$sbatch_folder = $ARGV[2]; # .../pssm/nr90/sbatch
$nr_db = $ARGV[3]; # nr90


opendir(DIR,"$fastadir") || die "Faild to open directory $fastadir\n";
@files = readdir(DIR);
closedir(DIR);

$c=0;
foreach $file (@files)
{
  chomp $file;

  if($file eq '.' or $file eq '..' or index($file,'.fasta') <0)
  {
    next;
  }

  $c++;

  $fastafile = "$fastadir/$file";
  if(!(-e $fastafile))
  {
    die "Failed to find fasta file $fastafile\n";
  }

  if(!(-d $sbatch_folder))
  {
    `mkdir $sbatch_folder`;
  }

  $targetid = substr($file,0,index($file,'.fasta'));
  if(!(-d "$outputdir/$targetid"))
  {
    `mkdir $outputdir/$targetid`;
  }

  print "\n\n###########  processing $file\n";

  $runfile="$sbatch_folder/$targetid-jackhm.sh";
  print "Generating $runfile\n";
  open(SH,">$runfile") || die "Failed to write $runfile\n";

  print SH "#!/bin/bash -l\n";
  print SH "#SBATCH -J  $targetid\n";
  print SH "#SBATCH -o $targetid.out\n";
  print SH "#SBATCH -p Interactive\n";
  print SH "#SBATCH -N 1\n";
  print SH "#SBATCH -n 4\n"; #modified previous:1
  print SH "#SBATCH --mem 20G\n"; #modified previous:10G
  print SH "#SBATCH -t 04:00:00\n\n\n";


  # Commands here run only on the first core
  print SH "echo \"Running program on \$(hostname), reporting for duty.\"\n\n";
  print SH "echo \" perl /storage/htc/bdm/DNCON4/feature/aln/scripts/generate_jackhm_aln.pl $fastafile $nr_db  $outputdir/$targetid &> $outputdir/$targetid.log\"\n\n";

  print SH "SECONDS=0\n\n";

  print SH "perl /storage/htc/bdm/DNCON4/feature/aln/scripts/generate_jackhm_aln.pl $fastafile $nr_db  $outputdir/$targetid &> $outputdir/$targetid.log\n\n";

  print SH "duration=\$SECONDS\n\n";
  print SH "echo \"\$((\$duration / 60)) minutes and \$((\$duration % 60)) seconds elapsed.\"\n\n";

  close SH;

}
