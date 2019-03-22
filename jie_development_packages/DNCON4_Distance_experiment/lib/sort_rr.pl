$num = @ARGV;
if($num !=2)
{
	die "The parameter is not correct!\n";
}

$infile = $ARGV[0];
$outputfile = $ARGV[1];

open(IN,"$infile") || die "Failed to open file $infile\n";
open(OUT,">$outputfile") || die "Failed to open file $outputfile\n";

%sort_hash = {};
while(<IN>)
{
	$line=$_;
	chomp $line;
	@array = split(/\s+/,$line);
	$prob = pop @array;
	$sort_hash{$line} = $prob;
}
close IN;

foreach  $line (sort { $sort_hash{$b} <=> $sort_hash{$a} } keys %sort_hash) {
    print OUT "$line\n";
}
close OUT;

