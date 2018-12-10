use strict;
use warnings;
use File::Basename;
use Carp;

my $rr_dir = shift; #"/home/siva/Documents/programs/CNN/CAMEO_test/casp12fm/nr90_rr";
my $pdb_dir = shift; #"/home/siva/Documents/programs/CNN/CAMEO_test/casp12fm/target";

my @pdb_dir = glob "$pdb_dir/*.pdb";
#my @rr_files = glob "$rr_dir/*.dncon2.rr";
my @rr_files = glob "$rr_dir/*";
my $rr_file;

foreach my $file(@pdb_dir){
    my $pdb_file = $file;
    my $pdb = basename($file);
    $pdb =substr($pdb,0,index($pdb,'.pdb'));
    my @spl = split('-', $pdb);
    my $id = $spl[0];

    foreach $file (@rr_files){
      my $seq_id= basename($file);

      if($seq_id =~ /^$id/){
            #print $pdb_file."\n";
            #print $file."\n";
            print `perl /storage/htc/bdm/Collaboration/jh7x3/Contact_prediction_with_Tianqi/DNCON3/scripts/coneva-lite.pl -rr $file -pdb $pdb_file`;
         }
      }
    }

    # my $dist_path=join("",$dist_dir,$seq_name,".dist");
    # print $rr_file," ", $dist_path," ",$length," ",$seq,"\n";
