#perl organize_repeated_run_results.pl  /storage/hpc/scratch/zggc9/DNSS2/models/CRMN1Dconv_ss/self_ensemble_results /storage/hpc/scratch/zggc9/DNSS2/models/CRMN1Dconv_ss/self_ensemble_results/model_summary/ 10
#
$num = @ARGV;
if($num != 3)
{
	die "The number of parameter is not correct!\n";
}

$inputdir = $ARGV[0];
$outputdir = "$ARGV[1]";
$repeated_num = $ARGV[2];

if(!(-d $outputdir))
{
	`mkdir $outputdir`;
}

open(OUT,">$outputdir/model.list") || die "Failed to write $outputdir/model.list\n";

for($i=1;$i<=$repeated_num;$i++)
{
	$modeldir = "$inputdir/$i";
	if(!(-d $modeldir))
	{
		die "Failed to find directory $modeldir\n";
	}
	opendir(SUBDIR,"$modeldir") || die "Failed to find directo$modeldir\n";
	@subdirs = readdir(SUBDIR);
	closedir(SUBDIR);
	$net_indx = 0;
	foreach $dir (@subdirs)
	{
		chomp $dir;
		if($dir eq '.' or $dir eq '..')
		{
			next;
		}
		$net_indx++;
		$model_outdir = "$modeldir/$dir";
		opendir(SUBDIR2,"$model_outdir") || die "Failed to find directory $model_outdir\n";
		@subdirs2 = readdir(SUBDIR2);
		closedir(SUBDIR2);
		foreach $file (@subdirs2)
		{
			chomp $file;
			if($file eq '.' or $file eq '..')
			{
				next;
			}

			if(index($file,'.json')>0)
			{
				$modelname = substr($file,index($file,'deepss'),index($file,'.json')-index($file,'deepss'));
				$modelfile = "$model_outdir/$file";
				print "Found $modelfile with $modelname\n";
				print "cp $modelfile $outputdir/model-train-$modelname-$i-$net_indx.json\n";
				`cp $modelfile $outputdir/model-train-$modelname-$i-$net_indx.json`;
				print OUT "$modelname-$i-$net_indx\n";
			}

			if(index($file,'-best-val.h5')>0)
			{
				$modelname = substr($file,index($file,'deepss'),index($file,'-best-val')-index($file,'deepss'));
				$modelfile = "$model_outdir/$file";
				print "Found $modelfile with $modelname\n";
				print "cp $modelfile $outputdir/model-train-weight-$modelname-$i-$net_indx-best-val.h5\n";
        		`cp $modelfile $outputdir/model-train-weight-$modelname-$i-$net_indx-best-val.h5`;
			}

		}
		print "\n";
	}
}
close OUT;
