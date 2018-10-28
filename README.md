# DNCON4
Designing and benchmarking deep learning architectures for protein contact prediction




### Data set construction
--------------------------------------------------------------------------------------

**(A) Download cullpdb list for standard training (pairwise-similarity: 30%, Resolution: <2.5, R: <1.0)**  
```
*** Details: http://dunbrack.fccc.edu/Guoli/culledpdb_hh/

cd /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/
wget http://dunbrack.fccc.edu/Guoli/culledpdb_hh/cullpdb_pc30_res2.5_R1.0_d181018_chains15535.gz
wget http://dunbrack.fccc.edu/Guoli/culledpdb_hh/cullpdb_pc30_res2.5_R1.0_d181018_chains15535.fasta.gz
gzip -d cullpdb_pc30_res2.5_R1.0_d181018_chains15535.fasta.gz
gzip -d cullpdb_pc30_res2.5_R1.0_d181018_chains15535.gz

*** contains 15535 proteins
```

**(B) Download pdb file from protein data bank**
```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P1_download_pdb_for_train_cullpdb.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_chains15535 /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/NewTrainTest_20181027/scripts/  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb 

*** 48 chians failed to find pdb (4Y40, 4YBB,4v4e,4v4m,4v9f,5XLI-B,5XLI-C)
```

**(C) Summary the pdb information for dataset**
```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/N1_download_pdb_for_train_cullpdb_summary.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/cullpdb_pc30_res2.5_R1.0_d181018_chains15535 /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed.summary  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed.fasta

*** total remaining proteins are: 15487
```


**(D) ### Remove proteins with chain-break (Ca-Ca distance > 4 angstrom)**
```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P3_remove_protein_by_CA_CA_distance.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed.fasta /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta 

*** remaining proteins have 8026, saved in /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta

*** removed protein saved in /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta.log
```

**(E) ### Organize fasta directory**
```
mkdir /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/fasta
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P4_extract_fastas.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta    /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/fasta
```

**(F) ### Organize chain directory**
```
mkdir /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains/
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/N2_copy_file.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains/  chn
```

**(F) ### Check if all files are ready**
```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/N2_check_file_existence.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/fasta/  fasta

perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/N2_check_file_existence.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains/  chn
```

**(G) ### Statistics (8026 proteins)**
```
Processed dataset: /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed-CACA4.fasta   ---- 8026 proteins

Information of proteins, including PDBcode, Resolution, X_ray, R-factor, FreeRvalue, ReleaseDate
/storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc30_res2.5_R1.0_d181018_processed.summary
```

**(H) ### Further criteria **
***1. Length***
***2. Release Date***
