# DNCON4
Designing and benchmarking deep learning architectures for protein contact prediction


**Attention**
1. Please don't add folder '/storage/htc/bdm/DNCON4/data' into git log system, it is large

### Data set construction
--------------------------------------------------------------------------------------

**(A) Download cullpdb list for standard training (pairwise-similarity: 25%, Resolution: <2.5, R: <1.0)**  
```
*** Details: http://dunbrack.fccc.edu/Guoli/culledpdb_hh/

cd /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/
wget http://dunbrack.fccc.edu/Guoli/culledpdb_hh/cullpdb_pc25_res2.5_R1.0_d181025_chains12572.gz
wget http://dunbrack.fccc.edu/Guoli/culledpdb_hh/cullpdb_pc25_res2.5_R1.0_d181025_chains12572.fasta.gz
gzip -d cullpdb_pc25_res2.5_R1.0_d181025_chains12572.fasta.gz
gzip -d cullpdb_pc25_res2.5_R1.0_d181025_chains12572.gz

*** contains 12566 proteins
```

**(B) Download pdb file from protein data bank**
```
#perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P1_download_pdb_for_train_cullpdb.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181025_chains12572 /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/NewTrainTest_20181027/scripts/  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb 

cd /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/scripts
sbatch P1_download_pdb.sh

*** 36 chians failed to find pdb (4Y40, 4YBB,4v4e,4v4m,4v9f,5XLI-B,5XLI-C)
```

**(C) Summary of the pdb information for dataset**
```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/N1_download_pdb_for_train_cullpdb_summary.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181025_chains12572 /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181025_processed.summary  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed.fasta

*** total remaining proteins are: 12530
```


**(D) ### Remove proteins with chain-break (Ca-Ca distance > 4 angstrom) and protein contains non-standard amino acid**
```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P3_remove_protein_by_CA_CA_distance.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed.fasta /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta 

*** remaining proteins have 6570, saved in /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta 

*** removed protein saved in /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta.log
```

**(E) ### Reindex pdb and Convert chain seq 2 fasta format**

mkdir /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P4_reindex_pdb_batch.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/orig_pdb /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains/  chn

```
mkdir /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/fasta
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P4_get_seq_from_pdb_batch.pl  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta    /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/fasta chn
```


**(F) ### convert pdb file to dssp file 

```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P5_pdb2dssp_bylist.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/scripts/dssp /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/dssp/  /scratch/jh7x3/tools/dssp


*** check if all dssp have been generated
perl /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/NewTrainTest_20181027/scripts/N2_check_file_existence.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/dssp/  dssp
```


**(G) ### Check if the sequence of pdb, dssp, and fasta sequence are same, with length 26-700**
```
perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P6_check_dssp_pdb_fasta.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4.fasta /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/chains/ /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/dssp/ /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/fasta/ /storage/htc/bdm/Collaboration/Zhiye/SSP/DNSS2/datasets/NewTrainTest_20181027/scripts/dssp2dataset_dnss.pl  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4-examined.fasta

```


**(G) ### Statistics (6085 proteins)**
```
Processed dataset
/storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4-examined.fasta    
---- 6085 proteins


Information of proteins, including PDBcode, Resolution, X_ray, R-factor, FreeRvalue, ReleaseDate
/storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4-examined.summary
```

**(H) ### organize the data for training 

```
cd /storage/htc/bdm/DNCON4/data/cullpdb_dataset
mkdir  fasta dssp chains lists
cp /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4-examined.fasta  /storage/htc/bdm/DNCON4/data/cullpdb_dataset/lists/all.fasta
cp /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/data_processing/cullpdb_pc25_res2.5_R1.0_d181018_processed-CACA4-examined.summary /storage/htc/bdm/DNCON4/data/cullpdb_dataset/lists/all.fasta.info


perl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/scripts/P8_summarize_data_files.pl /storage/htc/bdm/DNCON4/data/cullpdb_dataset/lists/all.fasta /storage/htc/bdm/DNCON4/data/cullpdb_dataset/source_data/ /storage/htc/bdm/DNCON4/data/cullpdb_dataset
```



