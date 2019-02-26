use strict;
use warnings;
use File::Basename;

my $path = shift;
my $output_file = shift;
my @allFiles = glob("$path/*");

open NEFF, ">>$output_file" or confess $!;

chdir("$path");
foreach my $file(@allFiles)
 {

    print "$file\n";
	  next if ($file eq '.' or $file eq '..' or index($file,'.aln') <0);
    my $id = substr($file,0,index($file,'.aln'));
	if(! -f "$id.colstats"){
		    `/storage/htc/bdm/tools/MetaPSICOV/bin/alnstats $id.aln $id.colstats $id.pairstats`;
	}
    open COLSTATS, "<$id.colstats" or confess $!;
    while(<COLSTATS>){
    	my $jackaln_neff=<COLSTATS>;
    	chomp($jackaln_neff);
    	print NEFF "$id N:$jackaln_neff ";
    	$jackaln_neff =<COLSTATS>;
    	chomp($jackaln_neff);
    	print NEFF " Neff:$jackaln_neff\n";
    	last;
    }

 }

close NEFF;
