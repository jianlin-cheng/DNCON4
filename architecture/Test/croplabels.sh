#!/bin/bash -
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 512
#SBATCH -p General
#SBATCH -J python2.7_virtenv_numpy_example
#SBATCH -o python2.7_virtenv_numpy_example_%j.out

## Activate the Virtual Environment
source ~/FQ3.6/bin/activate

## Execute our Numpy Example
python CropLabels.py labels.lst

## DeActivate the Virtual Environment
deactivate
