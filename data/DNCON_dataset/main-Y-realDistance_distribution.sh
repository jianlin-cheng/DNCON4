#!/bin/bash

dir_root=/home/jh7x3/scratch/DNCON2_train/input-data-for-training/test-train

cat $dir_root/lists/test.lst | while read id; do
	echo $id
	./generate-Y-DistanceDistribution.pl $dir_root/fastas/$id.fasta $dir_root/dist/$id.dist  > $dir_root/target_RealDistanceDistribution/Y-realDistDistribution-$id.txt
done

cat $dir_root/lists/train.lst | while read id; do
	echo $id
	./generate-Y-DistanceDistribution.pl $dir_root/fastas/$id.fasta $dir_root/dist/$id.dist  > $dir_root/target_RealDistanceDistribution/Y-realDistDistribution-$id.txt
done
