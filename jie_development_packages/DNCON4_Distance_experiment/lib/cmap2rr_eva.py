# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:23:17 2019

@author: Tianqi
"""
import sys
import subprocess
import os,glob,re

from math import sqrt

import numpy as np

if len(sys.argv) != 5:
    print('####please input the right parameters####')
    sys.exit(1)
# Locations of .cmap
cmap_file = sys.argv[1]
fasta_file = sys.argv[2]
rr_file = sys.argv[3]
pdb_file = sys.argv[4]
# ################## Download and prepare the dataset ##################


f = open(rr_file+".raw",'w')
cmap = np.loadtxt(cmap_file,dtype='float32')
L = cmap.shape[0]
for i in range(0,L):
   for j in range(i+1,L):
       f.write(str(i+1)+" "+str(j+1)+" 0 8 "+str(cmap[i][j])+"\n")
f.close()

subprocess.call("perl /scratch/jh7x3/DNCON4/jie_development_packages/DNCON4_Distance_experiment/lib/sort_rr.pl  "+rr_file+".raw  "+ rr_file+".raw.sorted",shell=True)

os.system('egrep -v \"^>\" '+fasta_file+'  > '+rr_file)
os.system('cat '+rr_file+'.raw.sorted >> '+rr_file)
os.system('rm -f '+rr_file+'.raw')
os.system('rm -f '+rr_file+'.raw.sorted')


subprocess.call("perl /storage/htc/bdm/DNCON4/test/scripts/coneva-lite.pl -rr "+rr_file+" -pdb "+ pdb_file+" >> "+rr_file+"-eva.txt",shell=True)
print("PRECISION                     Top-5     Top-L/10  Top-L/5   Top-L/2   Top-L     Top-2L    ")
for line in open(rr_file+"-eva.txt",'r'):
    line = line.rstrip()
    if(".pdb (precision)" in line):
        arr = line.split()
        print(arr[0])
    if(".rr (precision)" in line):
        print(line),

