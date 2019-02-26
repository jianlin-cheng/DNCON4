#!/usr/bin/perl -w

$numArgs = @ARGV;
if($numArgs != 1)
{
	print "the number of parameters is not correct!\n";
	exit(1);
}

$wordir		= "$ARGV[0]"; #/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/results/example_sbatch/
$i=0;
chdir($wordir);

@files = glob("./*.sh");

foreach $file(@files){
       print "Runnung $file\n";
	   $i++;
	       if($i % 100==0){
				   print "let's wait 5s......\n";
				   sleep(5);
            }
      `sbatch $file`;
}
