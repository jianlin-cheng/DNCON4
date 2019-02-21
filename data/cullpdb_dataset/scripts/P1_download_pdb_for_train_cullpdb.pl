 #! /usr/bin/perl -w
 # 
 
$num = @ARGV;
if($num != 3)
{
  die "The number of parameter is not correct!\n";
}

$pdb_list = $ARGV[0];
$script_dir = $ARGV[1];
$outputdir = $ARGV[2];

-f "$script_dir/getres.pl" || die "Failed to find $script_dir/getres.pl\n";
-f "$script_dir/autoftp" || die "Failed to find $script_dir/autoftp\n"; #/storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/NewTrainTest_20181027/scripts/autoftp

if(!(-d $outputdir))
{
  `mkdir $outputdir`;
}
chdir($outputdir);
if(-e "$outputdir/chain_resolution.txt")
{
  `cp $outputdir/chain_resolution.txt $outputdir/chain_resolution_tmp.txt`;
}
if(-e "$outputdir/all.fasta")
{
  `cp $outputdir/all.fasta $outputdir/all_tmp.fasta`;
}
open(LOG,">>$outputdir/chain_resolution.txt") || die "Failed to open file $outputdir/chain_resolution.txt\n";
open(FASTA,">>$outputdir/all.fasta") || die "Failed to open file $outputdir/all.fasta\n";
open(IN,$pdb_list) || die "Failed to open file $pdb_list\n";
print LOG "PDBcode Resolution X_ray R-factor FreeRvalue ReleaseDate\n";

$c=0;
while(<IN>)
{
  $line=$_;
  chomp $line;
  if(index($line,'IDs')>=0)
  {
    next;
  }
  @array = split(/\s++/,$line);
  $protein = $array[0];
  if(length($protein) !=5)
  {
    print "The format of $protein is not correct!\n";
    next;
  }
  $pdbname = lc(substr($protein,0,4));
  $pdbname_uc = uc(substr($protein,0,4));
  $chainid = substr($protein,4);
  if(-e "$pdbname_uc-$chainid.chn" and -e "$pdbname_uc-$chainid.fasta")
  {
    next;
  }
  
  ## download pdb 
  
  `$script_dir/autoftp -u anonymous -p anonymous 'ftp.wwpdb.org;./pub/pdb/data/structures/all/pdb;b;pdb$pdbname.ent.gz'`;
  if(!(-e "pdb$pdbname.ent.gz"))
  {
    print "Failed to download pdb$pdbname.ent.gz\n";
    next;
  }
  `gzip -d pdb$pdbname.ent.gz`;
  
  
  #### get resolution from pdb
  open(PDB, "pdb$pdbname.ent") || die "fail to open unzip pdb file:pdb$pdbname.ent.\n";
  @content = <PDB>;
  close PDB;
  
  $x_ray = "O"; #default: non-x-ray 
  $resolution = "6.0";
  $release_date = "UNKNOWN";
  $rfactor = $array[4];
  $freeRvalue = $array[5];
  
  foreach $text(@content)
  {
  	chomp $text; 
  	#check if it is X-RAY
  	if ($text =~ /.*EXPDTA\s+X-RAY/)
  	{
  		$x_ray = "X"; 
  	}
  	#extract resolution
  	if ($text=~/.*REMARK.*2.*RESOLUTION.*ANGSTROMS.*/)
  	{
  	     ($tmp, $tmp, $tmp, $resolution, @otherstuff) = split(/\s+/, $text);
  	     @otherstuff = (); 
  	     $resolution =~ /^(\d+\.\d+)/; 
  	     $resolution = $1; 
  	     if (length($resolution) == 0)
  	     {
  	     	#some case, not resolution information (not applicable), set resolution to 10A. 
  	    	$resolution = 6.0; 
  	     	print "$pdbname: resolution is not found, set to 6.0\n"; 
  	     }
  	}
    if ($text=~/.*REVDAT.*1.*/)
  	{
  	     ($tmp, $tmp, $release_date, @otherstuff) = split(/\s+/, $text);
  	     @otherstuff = (); 
  	     if (length($release_date) == 0)
  	     {
  	     	#some case, not resolution information (not applicable), set resolution to 10A. 
  	    	$release_date = "None"; 
  	     	print "$pdbname: release date is not found, set to UNKNOWN\n"; 
  	     }
  	}
  }
  $c++;
  print "$c: $pdbname_uc-$chainid $resolution $x_ray $rfactor $freeRvalue $release_date\n";
  print LOG "$pdbname_uc-$chainid $resolution $x_ray $rfactor $freeRvalue $release_date\n";
  
  ## get chain pdb 
  $file_PDB = "pdb$pdbname.ent";
  open INPUTPDB, $file_PDB or die "ERROR! Could not open $file_PDB";
  my @lines_PDB = <INPUTPDB>;
  close INPUTPDB;
  
  
  open(OUT,">$pdbname_uc-$chainid.chn") || die "Failed to open file $pdbname_uc-$chainid.chn\n"; #1IO0-A.chn
  foreach (@lines_PDB) {
  	next if $_ !~ m/^ATOM/;
    $this_chain = parse_pdb_row($_,"chain");
    if($this_chain eq $chainid)
    {
      print OUT $_;
    }
  }
  close OUT;
  
  ## get original chain fasta 
  `curl -X POST  -o $pdbname_uc-$chainid.tmp --data "fileFormat=fastachain&compression=NO&structureId=$pdbname&chainId=$chainid" https://www.rcsb.org/pdb/download/downloadFile.do &>log.txt`;
  
  if(!(-e "$pdbname_uc-$chainid.tmp"))
  {
    print "Failed to download $pdbname_uc-$chainid.tmp, try lower case\n";
    $chainid_lc = lc($chainid);
    `curl -X POST  -o $pdbname_uc-$chainid.tmp --data "fileFormat=fastachain&compression=NO&structureId=$pdbname&chainId=$chainid_lc" https://www.rcsb.org/pdb/download/downloadFile.do &>log.txt`;
  }
  if(!(-e "$pdbname_uc-$chainid.tmp"))
  {
    print "Failed to download $pdbname_uc-$chainid.tmp, next\n";
    next;
  }
  open(SEQ, "$pdbname_uc-$chainid.tmp") || die "fail to open $pdbname_uc-$chainid.tmp.\n";
  @content = <SEQ>;
  close SEQ;
  open(SEQOUT, ">$pdbname_uc-$chainid.fasta") || die "fail to open $pdbname_uc-$chainid.fasta.\n";
  foreach $li (@content)
  {
    chomp $li;
    if(substr($li,0,1) eq '>')
    {
      print FASTA uc($li)."\n";
      print SEQOUT uc($li)."\n";
    }else{
      print FASTA "$li";
      print SEQOUT "$li";
    }
  }
  print FASTA "\n";
  print SEQOUT "\n";
  `rm $pdbname_uc-$chainid.tmp`;
  `rm pdb$pdbname.ent`;
  
}
close IN;
close LOG;
close FASTA;


sub parse_pdb_row{
	my $row = shift;
	my $param = shift;
	my $result;
	$result = substr($row,6,5) if ($param eq "anum");
	$result = substr($row,12,4) if ($param eq "aname");
	$result = substr($row,16,1) if ($param eq "altloc");
	$result = substr($row,17,3) if ($param eq "rname");
	$result = substr($row,22,5) if ($param eq "rnum");
	$result = substr($row,21,1) if ($param eq "chain");
	$result = substr($row,30,8) if ($param eq "x");
	$result = substr($row,38,8) if ($param eq "y");
	$result = substr($row,46,8) if ($param eq "z");
	die "Invalid row[$row] or parameter[$param]" if (not defined $result);
	$result =~ s/\s+//g;
	return $result;
}
