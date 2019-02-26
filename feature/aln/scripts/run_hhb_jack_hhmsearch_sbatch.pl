#!/usr/bin/perl -w
# perl run_hhjack_hhmsearch_sbatch.pl /storage/htc/bdm/DNCON4/feature/fasta /storage/htc/bdm/DNCON4/feature/aln/hhb_jack_hhmsearch /storage/htc/bdm/DNCON4/feature/aln/hhb_jack_hhmsearch_sbatch

use POSIX;

if (@ARGV != 3 ) {
  print "Usage: <fasta> <outdir> <sbatch_dir>\n";
  exit;
}


$fastadir = $ARGV[0]; #/storage/htc/bdm/tools/DNCON2/fastas/
$outputdir = $ARGV[1]; #.../pssm/nr90/
$sbatch_folder = $ARGV[2]; # .../pssm/nr90/sbatch


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

  $runfile="$sbatch_folder/$targetid-hhb_jack_hhmsearch.sh";
  print "Generating $runfile\n";
  open(SH,">$runfile") || die "Failed to write $runfile\n";

  print SH "#!/bin/bash -l\n";
  print SH "#SBATCH -J  $targetid\n";
  print SH "#SBATCH -o $targetid.out\n";
  print SH "#SBATCH -p Lewis,hpc5\n";
  print SH "#SBATCH -N 1\n";
  print SH "#SBATCH -n 8\n"; #modified previous:1
  print SH "#SBATCH --mem 20G\n"; #modified previous:10G
  print SH "#SBATCH -t 2-00:00:00\n\n\n";


  # Commands here run only on the first core
  print SH "echo \"Running program on \$(hostname), reporting for duty.\"\n\n";

  print SH "SECONDS=0\n\n";
  print SH "export HHLIB=/storage/htc/bdm/tools/hhsuite-2.0.16-linux-x86_64/lib/hh\n\n";
  print SH "PATH=\$PATH:/storage/htc/bdm/tools/hhsuite-2.0.16-linux-x86_64/bin:\$HHLIB/scripts\n\n";
  print SH "perl /storage/htc/bdm/DNCON4/feature/aln/scripts/generate_hhjack_hhmsearch3.pl $fastafile $outputdir/$targetid &> $outputdir/$targetid.log\n\n";

  print SH "duration=\$SECONDS\n\n";
  print SH "echo \"\$((\$duration / 60)) minutes and \$((\$duration % 60)) seconds elapsed.\"\n\n";

  close SH;

}
