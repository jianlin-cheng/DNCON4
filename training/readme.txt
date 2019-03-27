This directory is used to train and optimize deep learning architectures.
1. combine domain contact prediction with full-length contacts
First step reindex domain contact based on the "domain_info"
perl dncon2-domain.pl <full-length fasta file> <domain folder that contains domain rr raw files>

Second step, compare domain contacts with full-length, replace lower contact scores
perl contact_combine.pl <full-length rr raw file> <domain folder that contains domain reindex rr raw files> <outdir for the combined rr >

sample:
cd /storage/htc/bdm/DNCON4/training/domain_combine/script
perl dncon2-domain.pl /storage/htc/bdm/tianqi/DNCON2/casp13/fasta/T0981.fasta /storage/htc/bdm/DNCON4/training/domain_combine/sample/T0981/dm/

perl contact_combine.pl /storage/htc/bdm/DNCON4/training/domain_combine/sample/T0981/fl/T0981.rr /storage/htc/bdm/DNCON4/training/domain_combine/sample/T0981/dm/ /storage/htc/bdm/DNCON4/training/domain_combine/sample/T0981/
