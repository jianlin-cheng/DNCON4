 #! /usr/bin/perl -w
 # perl fill_aln_gap_withX.pl  T0658.aln  T0658_new.fasta

$num = @ARGV;
if($num != 2)
{
  die "The number of parameter is not correct!\n";
}

$in_aln = $ARGV[0];
$out_aln = $ARGV[1];


open(IN,$in_aln) || die "Failed to open file $in_aln\n";
open(OUT,">${in_aln}_tmp.fa") || die "Failed to open file ${in_aln}_tmp.fa\n";
@content = <IN>;
close IN;

$orig_seq= shift @content;
chomp $orig_seq;

$ind =0;
foreach (@content)
{
  $line=$_;
  chomp $line;
  #if(index($line,'X') >= 0)
  #{
  #  die "!!! the sequence should not have X\n\n";
  #}
  #$line =~ s/\-/X/g;
  $ind++;
  print OUT ">aln_$ind\n$line\n";
}
close IN;
close OUT;



#### use CD-HIT to filter
`rm -rf *_tmp_DB*`;

#`/storage/htc/bdm/tools/cd-hit-v4.6.8-2017-1208/cd-hit -i ${in_aln}_tmp.fa -o ${in_aln}_tmp_DB_50_seq.fasta -G 0 -aS 0.5 -n 3 -d 0 -M 0 -T 0`;
`/storage/htc/bdm/tools/cd-hit-v4.6.8-2017-1208/cd-hit-auxtools/cd-hit-dup -i ${in_aln}_tmp.fa -o ${in_aln}_tmp_DB_50_seq.fasta -e 0.01`;

open(OUT,">$out_aln") || die "Failed to open file $out_aln\n";
print OUT "$orig_seq\n";
open(IN,"${in_aln}_tmp_DB_50_seq.fasta") || die "Failed to open file ${in_aln}_tmp_DB_50_seq.fasta\n";
while(<IN>)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
    next;
  }

  print OUT "$line\n";
}

close OUT;

`rm -rf  ${in_aln}_tmp*`;
