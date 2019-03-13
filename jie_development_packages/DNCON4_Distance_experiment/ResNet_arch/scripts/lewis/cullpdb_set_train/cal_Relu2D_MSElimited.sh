#!/bin/bash
#--------------------------------------------------------------------------------
#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH -J deepcov         # name for the job
#SBATCH -o ResNet-MSEloop-%j.out
#SBATCH -N 1                   # number of cores
#SBATCH -n 1
#SBATCH --mem=2G                           # total memory
#SBATCH --time 2-00:00:00                   # time limit in the form days-hours:minutes

#SBATCH --partition Lewis,hpc5
#--------------------------------------------------------------------------------
cd /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/ResNet_arch/scripts/lewis/cullpdb_set_train
sh ./cal_Relu2D_MSElimited_inloop.sh
