$num= @ARGV;
if($num != 5)
{
	die "The number of parameter is not correct!\n";
}

$contactdir = $ARGV[0];
$fastadir = $ARGV[1];
$domaindir = $ARGV[2];
$inputlist = $ARGV[3];
$outputfile = $ARGV[4];



#python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/cmap2rr_eva.py evadir/pred_map/T0949-contact.txt  /storage/htc/bdm/DNCON4/test/CASP13/fasta/T0949.fasta  evadir/pred_map/T0949-contact.rr /storage/htc/bdm/DNCON4/test/CASP13/pdb/T0949-D1.pdb >> evadir/casp13_eva.txt

opendir(DIR,"$domaindir") || die "Failed to open file $domaindir\n";
@files = readdir(DIR);
closedir(DIR);

open(IN,"$inputlist") || die "Failed to open file $inputlist\n";
%domainlist = ();
while(<IN>)
{
	$line=$_;
	chomp $line;
	#print "Loading $line: ".length($line)."\n";
	$domainlist{$line}=1
}
close IN;

open(OUT,">$outputfile") || die "Failed to open file $outputfile\n";
print OUT "DomainID\tTop5\tTopL10\tTopL5\tTopL2\tTopL\tTop2L\n";

$Top5_avg = 0;
$TopL10_avg = 0;
$TopL5_avg = 0;
$TopL2_avg = 0;
$TopL_avg = 0;
$Top2L_avg = 0;
$domain_num=0;
foreach $file ( sort @files)
{
	chomp $file;
	if($file eq '.' or $file eq '..')
	{
		next;
	}
	@array = split(/\./,$file);
	$domainid = $array[0];
	if(!exists($domainlist{$domainid}))
	{
		#print "Failed to find $domainid\n";
		next;
	}
	@array2 = split(/\-/,$domainid);
	if(@array2 !=2)
	{
		next;
	}
	$targetid = $array2[0];
	if(!(-e "$contactdir/$targetid-contact.txt"))
	{
		print "Failed to find $contactdir/$targetid-contact.txt\n";
		next;
	}
	
	if(!(-e "$fastadir/$targetid.fasta"))
	{
		print "Failed to find $fastadir/$targetid.fasta\n";
		next;
	}
	`rm $contactdir/$domainid-contact.rr-eva.txt`;
	#print "python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/cmap2rr_eva.py $contactdir/$targetid-contact.txt $fastadir/$targetid.fasta $contactdir/$domainid-contact.rr $domaindir/$file > $contactdir/$domainid-contact.eva\n\n";
	system("python /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/cmap2rr_eva.py $contactdir/$targetid-contact.txt $fastadir/$targetid.fasta $contactdir/$domainid-contact.rr $domaindir/$file > $contactdir/$domainid-contact.eva");
	$domain_num++; 
	#print "Evaluating $domainid\n";
	open(IN,"$contactdir/$domainid-contact.eva") || die "Failed to open file $contactdir/$domainid-contact.eva\n";
	print "open $contactdir/$domainid-contact.eva\n";
	while(<IN>)
	{
		$line = $_;
		chomp $line;
		if(index($line,'(precision)')<=0)
		{
			next;
		}
		$scores = substr($line,index($line,'(precision)')+length('(precision)'));
		@score_array = split(/\s+/,$scores);
		$Top5 = $score_array[0];     
		$TopL10 = $score_array[1];  
		$TopL5 = $score_array[2];   
		$TopL2 = $score_array[3];   
		$TopL = $score_array[4];    
		$Top2L = $score_array[5];
		
		$Top5_avg += $Top5;
		$TopL10_avg += $TopL10;
		$TopL5_avg += $TopL5;
		$TopL2_avg += $TopL2;
		$TopL_avg += $TopL;
		$Top2L_avg += $Top2L;
		print OUT "$domainid\t$Top5\t$TopL10\t$TopL5\t$TopL2\t$TopL\t$Top2L\n";
	} 
	close IN;
}


$Top5_avg /= $domain_num;
$TopL10_avg /= $domain_num;
$TopL5_avg /= $domain_num;
$TopL2_avg /= $domain_num;
$TopL_avg /= $domain_num;
$Top2L_avg /= $domain_num;

print OUT "Average($domain_num)\t$Top5_avg\t$TopL10_avg\t$TopL5_avg\t$TopL2_avg\t$TopL_avg\t$Top2L_avg\n";
close OUT;

