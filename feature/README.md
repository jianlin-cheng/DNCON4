# DNCON4 feature and label generation
This directory includes the scripts used to create the input feature sets for deep learning and the feature sets. 
### ALN
--------------------------------------------------------------------------------------

**(A) COVARIANCE MATRIX**  
```
cull_pdb: /storage/htc/bdm/DNCON4/feature/cov/cullpdb_cov/output(using hhblits_jack to genereate aln)
dncon2 : /storage/htc/bdm/DNCON4/feature/cov/DNCON2Data/cov (using Badri's original aln)
```

**(B) PLM MATRIX**
```
deepcov_plm: /storage/htc/bdm/DNCON4/feature/plm/deepcov_plm
dncon2: /storage/htc/bdm/DNCON4/feature/plm/dncon2_plm
```

**(C) Labels**
```
cullpdb: /storage/htc/bdm/DNCON4/feature/map/cullpdb_map 
dncon2: /storage/htc/bdm/DNCON4/feature/map/dncon2_map
where cmap is for binary contact classification
      dist_map is for 41-class classification
      real_dist_map is for real distance contact map
      
```

**(D) ALN**
```
CLUSTER_meta: /hhjack_hhmsearch3.sh <fasta> <out>
Example:
cd /data/commons/DNCON4_feature/aln/src
./hhjack_hhmsearch3.sh /data/commons/DNCON4_feature/aln/sample/T0953s2.fasta /data/commons/DNCON4_feature/aln/sample/T0953s2
```





