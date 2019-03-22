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
$c=0;
while(<IN>)
{
	$c++;
	$line=$_;
	chomp $line;
	#print "$c\n";
	@array = split(/\s+/,$line);
	$prob = pop @array;
	chomp $prob;
	$sort_hash{"$line"} = $prob;
	#print OUT "$line\n";
}
close IN;

foreach  $li (sort { $sort_hash{$a} <=> $sort_hash{$b} } keys %sort_hash) {
#foreach  $li (sort keys %sort_hash) {
    	@array = split(/\s/,$li);
	if(@array !=3)
	{
		next;
	}
	print OUT "$li\n";
}
close OUT;

