
## perl /storage/htc/bdm/tianqi/test/new_alignment/T0952/parse_hmmer3_fast.pl  /storage/htc/bdm/tianqi/test/new_alignment/T0952/T0952.out  /storage/htc/bdm/tianqi/test/new_alignment/T0952.fasta   /storage/htc/bdm/tianqi/test/new_alignment/T0952/T0952.fseqs.inline  /storage/htc/bdm/tianqi/test/new_alignment/T0952/


## perl /storage/htc/bdm/tianqi/test/new_alignment/T0952/parse_hmmer3_fast.pl  /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments/T0954-hmmsearch.out   /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments/T0954.fasta   /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments/T0954.feseqs.inline  /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments/


##perl /storage/htc/bdm/tianqi/test/new_alignment/T0952/parse_hmmer3_fast_V2.pl  /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments_jie/T0954-hmmsearch.out   /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments_jie/T0954.fasta   /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments_jie/T0954.feseqs.inline  /storage/htc/bdm/tianqi/DNCON3_combine_hhbjhm/casp13_gpu/output/T0954/fl/alignments_jie/



if (@ARGV != 5)
{
	die "need three parameters: option file, sequence file, output dir.\n";
}

$hmmer_outfile = shift @ARGV;
$fasta_file = shift @ARGV;
$hmmerdb  = shift @ARGV;
$work_dir = shift @ARGV;
$e = shift @ARGV;

#check fast file format
open(FASTA, $fasta_file) || die "can't read fasta file.\n";
$name_info = <FASTA>;
chomp $name_info;
$seq = <FASTA>;
chomp $seq;
close FASTA;
if ($name_info =~ /^>/)
{
	@name_s = split(/\s+/, $name_info);
	$name = $name_s[0];
	$name =~ s/>//;
}
else
{
	die "fasta foramt error.\n";
}
$targetid=$name;

####################################  option
#$hmmerdb = "/storage/htc/bdm/tools/uniref_updated/uniref90/uniref90.fasta";
if(!(-e $hmmerdb))
{
  die "Failed to find $hmmerdb\n\n";
}

$cm_blast_evalue = $e;

$hmmer3_dir="/storage/htc/bdm/tools/hmmer-3.1b2-linux-intel-x86_64/binaries/";
$script_dir="/storage/htc/bdm/DNCON4/feature/aln/scripts/";
####################################

if(!(-e "$script_dir/rank_templates.pl"))
{
  die "Failed to find $script_dir/rank_templates.pl\n\n";
}

if(!(-e "$script_dir/global2pir.pl"))
{
  die "Failed to find $script_dir/global2pir.pl\n\n";
}

if(!(-e "$script_dir/stock_2_fasta.pl"))
{
  die "Failed to find $script_dir/stock_2_fasta.pl\n\n";
}



#### process $hmmer_outfile, remove the description part
print "(1) start filter the hmmer output\n\n";
open(NEW,">$hmmer_outfile.filtered") || die "Failed to write $hmmer_outfile.filtered";
open(DIST,"$hmmer_outfile") || die "Failed to write $hmmer_outfile";
@dist = <DIST>;
close DIST;
$description_start=0;
while (@dist)
{
	$line = shift @dist;
	if ($line =~ /^\s+E-value\s+score\s+bias/)
	{
    $description_start = index($line,'Description');
    print NEW "$line";
		shift @dist;
		last;
	}
}

if($description_start==0)
{
  die "No templates found, exit\n\n";
}

while (@dist)
{
	$score = shift @dist;
	if ($score eq "\n")
	{
		last;
	}
	chomp $score;
  print NEW substr($score,0,$description_start)."\n";

}
close NEW;




#####################################################################################
#########################Stop here to work on rank_templates.pl######################
print "generate ranking list...\n";
print("$work_dir/$name.rank\n");
if(-e "$work_dir/$name.rank")
{
  print "$name.rank already generated\n\n";
} else {
	print "perl $script_dir/rank_templates.pl $hmmer_outfile.filtered $work_dir/$name.rank\n";
	system("perl $script_dir/rank_templates.pl $hmmer_outfile.filtered $work_dir/$name.rank");
}

#####################################################################################


###############################################################################
#select the templates whose evalue < $cm_blast_evalue (to do....) (an output fasta file)

open(RANK, "$work_dir/$name.rank") || die "can't open the template rank file.\n";
@rank = <RANK>;
close RANK;
shift @rank;

#return: -1: less, 0: equal, 1: more
sub comp_evalue
{
	my ($a, $b) = @_;
	#get format of the evalue
	if ( $a =~ /^[\d\.]+$/ )
	{
		$formata = "num";
	}
	elsif ($a =~ /^([\.\d]*)e([-\+]\d+)$/)
	{
		$formata = "exp";
		$a_prev = $1;
		$a_next = $2;
		if ($1 eq "")
		{
			$a_prev = 1;
		}
	#	if ($a_next > 0)
#		{
	#		die "exponent must be negative or 0: $a\n";
#		}
	}
	else
	{
		die "evalue format error: $a";
	}

	if ( $b =~ /^[\d\.]+$/ )
	{
		$formatb = "num";
	}
	elsif ($b =~ /^([\.\d]*)e([-\+]\d+)$/)
	{
		$formatb = "exp";
		$b_prev = $1;
		$b_next = $2;
		if ($1 eq "")
		{
			$b_prev = 1;
		}
	#	if ($b_next > 0)
	#	{
	#		die "exponent must be negative or 0: $b\n";
	#	}
	}
	else
	{
		die "evalue format error: $b";
	}
	if ($formata eq "num")
	{
		if ($formatb eq "num")
		{
			return $a <=> $b
		}
		else  #bug here
		{
			#a is bigger
			#return 1;
			#return $a <=> $b_prev * (10**$b_next);
			return $a <=> $b_prev * (10**$b_next);
		}
	}
	else
	{
		if ($formatb eq "num")
		{
			#a is smaller
			#return -1;
			#return $a_prev * (10 ** $a_next) <=> $b;
			return $a_prev * (10 ** $a_next) <=> $b;
		}
		else
		{
			if ($a_next < $b_next)
			{
				#a is smaller
				return -1;
			}
			elsif ($a_next > $b_next)
			{
				return 1;
			}
			else
			{
				return $a_prev <=> $b_prev;
			}
		}
	}
}

@sel_templates  = ();
@sel_evalues = ();

%selected_template_hash=();
while (@rank)
{
	$line = shift @rank;
	chomp $line;
	($index, $template, $evalue) = split(/\s+/, $line);
	$index = 0;
	if (&comp_evalue($evalue, $cm_blast_evalue) <= 0)
	{
		push @sel_templates, $template;
		push @sel_evalues, $evalue;
   $selected_template_hash{$template}=1;
	}

	$temp2evalue{$template} = $evalue;
}

if (@sel_templates <= 0)
{
    open(OUT,">$work_dir/$targetid.hmmer.msa") || die "Failed to open file $work_dir/$targetid.hmmer.msa\n";
	print OUT $seq."\n";
	print OUT $seq;
	close OUT;
	print "No templates with evalue than $cm_blast_evalue were found. Stop!\n";
	exit 0;
}

#print "Selected templates: @sel_templates\n";
print "Loading $hmmerdb\n\n";
open(HMMERDB, "$hmmerdb") || die "can't read $hmmerdb.\n";
#@allseq = <HMMERDB>;

$seq_read=0;
print "Start load $hmmerdb\n";
$found =0;
$found_num=0;
while (<HMMERDB>)
{
  $line = $_;
  chomp $line;

  if(substr($line, 0, 1) eq '>')
  {
    $temp_id = $line;
    if(index($temp_id,' ')>0)
    {
      @tmp = split(/\s/,$temp_id);
      $temp_id =$tmp[0];
    }
    $seq_read++;
    $temp_id = substr($temp_id, 1);
    if(exists($selected_template_hash{$temp_id}))
    {
      $found =1;
    }

    if($seq_read % 1000000 ==0)
    {
      #print "$seq_read ....\n";
    }
    next;

  }else{
    $temp_seq = $line;

    if( $found ==0)
    {
      next;
    }
    $id2seq{$temp_id} = $temp_seq;

      #print "found: $seq_read: $temp_id ....\n";
     $found_num++;


     if($found_num == @sel_templates)
     {
       print "All templates sequences have been found\n\n";
       last;
     }
     $found =0;
  }
}
close HMMERDB;
open(SEL, ">$name.sel") || die "can't create $name.sel.\n";

#first sequence is query itself
print SEL ">$name\n$seq\n";

foreach $sel_id (@sel_templates)
{
	if ( exists($id2seq{$sel_id}) )
	{
		$sel_seq = $id2seq{$sel_id};
		print SEL ">$sel_id\n$sel_seq\n";
	}
	else
	{
		warn "The sequence of $sel_id is not found.\n";
	}
}
close SEL;
###############################################################################
###############################################################################
#generate alignments between sam model and the selected sequence
#the alignment file is $name.a2m, which is a global alignment file
if(-e "$work_dir/$name.halign")
{
  print "$name.halign already generated\n\n";
}else{
  print("$hmmer3_dir/hmmalign $name.hmmer $name.sel > $name.halign\n\n");
  system("$hmmer3_dir/hmmalign $name.hmmer $name.sel > $name.halign");
}
#############################################################################


#convert alignment from stockhom format to fasta format
if(-e "$work_dir/$name.a2m")
{
	print "$name.a2m already generated\n\n";
}else{
	print("$script_dir/stock_2_fasta.pl $name.halign $name.a2m 50\n\n");
	system("$script_dir/stock_2_fasta.pl $name.halign $name.a2m 50");
	#convert sam global alignments to pir format
}


open(IN,"$name.a2m") || die "Failed to open file $name.a2m\n";
open(OUT,">$work_dir/$targetid.hmmer.msa") || die "Failed to open file $work_dir/$targetid.hmmer.msa\n";

$c=0;
$query_len=0;
$query_seq=0;
while(<IN>)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
      $c++;
      if($c>1)
      {
      	$seq = uc($seq);
      	#replace . with -
      	$seq =~ s/\./-/g;
        $tseq_len = length($seq);
        if($c==2)
        {
          $query_len = length($seq);
          $query_seq = $seq;
        }
        if($query_len != $tseq_len)
        {
          die "The length not match ($query_len != $tseq_len)\n\n";
        }

        $q_seq="";
        $t_seq="";
        for($i=0;$i<length($query_seq);$i++)
        {
          $q_res = substr($query_seq,$i,1);
          $t_res = substr($seq,$i,1);
          if($q_res ne '-')
          {
             $q_seq .=$q_res;
             $t_seq .=$t_res;
          }
        }

        if($c==2)
        {
          print OUT "$q_seq\n";
        }else{
          print OUT "$t_seq\n";
        }
      }

      $seq="";
  }else{
    $seq .= "$line";
  }
}
# process the last sequence
$seq = uc($seq);
#replace . with -
$seq =~ s/\./-/g;
$q_seq="";
$t_seq="";
for($i=0;$i<length($query_seq);$i++)
{
  $q_res = substr($query_seq,$i,1);
  $t_res = substr($seq,$i,1);
  if($q_res ne '-')
  {
     $q_seq .=$q_res;
     $t_seq .=$t_res;
  }
}

if($c==2)
{
  print OUT "$q_seq\n$t_seq\n";
}else{
  print OUT "$t_seq\n";
}
close OUT;

print "\n\nThe multiple sequence alignment is saved to $work_dir/$targetid.hmmer.msa\n\n";
