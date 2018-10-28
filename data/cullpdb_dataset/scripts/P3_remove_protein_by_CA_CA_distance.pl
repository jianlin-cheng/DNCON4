 #! /usr/bin/perl -w
 # 
 use Carp;
 use Scalar::Util qw(looks_like_number);
our %AA3TO1 = qw(ALA A ASN N CYS C GLN Q HIS H LEU L MET M PRO P THR T TYR Y ARG R ASP D GLU E GLY G ILE I LYS K PHE F SER S TRP W VAL V);
our %AA1TO3 = reverse %AA3TO1;
$num = @ARGV;
if($num != 3)
{
  die "The number of parameter is not correct!\n";
}

$sort30_file = $ARGV[0]; # /storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/selected_sort30_db/selected_sort30_from50-500_res2.5.fasta   
$atomdir = $ARGV[1]; #/storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON2_retrain_sort30/fastas
$outfile = $ARGV[2]; 



open(OUT,">$outfile") || die "Failed to open file $outfile\n";
open(ERR,">$outfile.log") || die "Failed to open file $outfile.log\n";

open(IN,$sort30_file) || die "Failed to open file $sort30_file\n";
@content = <IN>;
close IN;
$protein="";
foreach(@content)
{
  $line=$_;
  chomp $line;
  if(substr($line,0,1) eq '>')
  {
    $protein = substr($line,1);
    $protein_full = $protein;
    if(index($protein,'|')>0) #1A8L:A|PDBID|CHAIN|SEQUENCE
    {
      @tmp = split(/\|/,$protein);
      $protein = $tmp[0];
    }
  
    if(index($protein,' ')>0) #1A8L:A|PDBID|CHAIN|SEQUENCE
    {
      @tmp = split(/\s/,$protein);
      $protein = $tmp[0];
    }
    $protein =~ s/\:/\-/g;
    next;
  }else{

    
    $file_PDB = "$atomdir/$protein.chn";
    # Load PDB
    open INPUTPDB, $file_PDB or die "ERROR! Could not open $file_PDB";
    my @lines_PDB = <INPUTPDB>;
    close INPUTPDB;
    
    # (c) Get CA-CA atom
    print "Processing $file_PDB\n";
    %ca_hash = ();
    $seq = "";
    $res_id=0;
    $prev = -1;
    foreach (@lines_PDB) {
    	next if $_ !~ m/^ATOM/;
      $this_rnum = parse_pdb_row($_,"rnum");
    	$this_aname = parse_pdb_row($_,"aname");
      if($this_aname ne 'CA')
      {
        next;
      }
      
      if (!looks_like_number($this_rnum)) { #30A
        print "$this_rnum isn't a number in ".$_;
        next;
      }

			if ($this_rnum != $prev)
			{
      	$res = parse_pdb_row($_,"rname");
        if (exists($AA3TO1{$res}) )
        {
        	$res = $AA3TO1{$res}; 
        }
        else
        {
        	$res = "X"; 
        	print "$file_PDB: resudie is unknown, shouldn't happen.\n"; 
          last;
        }
				$seq .= $res; 
				$prev = $this_rnum; 
			
      	$this_x = parse_pdb_row($_,"x");
      	$this_y = parse_pdb_row($_,"y");
      	$this_z = parse_pdb_row($_,"z");
      	$this_x =~ s/^\s+|\s+$//g;
      	$this_y =~ s/^\s+|\s+$//g;
      	$this_z =~ s/^\s+|\s+$//g;
        $res_id++;
        $ca_hash{$res_id} = "$this_x $this_y $this_z";
      }
    }
    
    #if($line ne $seq) ###pdb seq doesn't need to be same as fasta seq
    #{
      #print "The fasta sequence not match pdb seq in $file_PDB\n$line\n$seq\n\n";
      #next;
    #}
    $found=0;
    for($res_id=2;$res_id<=length($seq);$res_id++)
    {
        $pos1 = $ca_hash{$res_id-1};
        $pos2 = $ca_hash{$res_id};
        @tmp1=split(/\s/,$pos1);
        @tmp2=split(/\s/,$pos2);
        $x1 = $tmp1[0];
        $y1 = $tmp1[1];
        $z1 = $tmp1[2];
        $x2 = $tmp2[0];
        $y2 = $tmp2[1];
        $z2 = $tmp2[2];
        $d = sqrt(($x1-$x2)**2+($y1-$y2)**2+($z1-$z2)**2);
        if($d>4)
        {
          $found=1;
          print ERR "$file_PDB has large CA-CA distance ($d) at position $res_id\n ";
          last;
        }
    }
    if($found==0)
    {
    
        print OUT ">$protein_full\n$line\n";
    
    }
  }
}
close OUT;
close ERR;

	



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