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

if len(sys.argv) != 4:
    print('####please input the right parameters####')
    sys.exit(1)
# Locations of .cmap
dist1_file = sys.argv[1]
dist2_file = sys.argv[2]
dist_out = sys.argv[3]
# ################## Download and prepare the dataset ##################

f = open(dist_out,'w')
cmap1 = np.loadtxt(dist1_file,dtype='float32')
cmap2 = np.loadtxt(dist2_file,dtype='float32')

errormap = np.absolute(cmap1-cmap2)
np.savetxt(dist_out,errormap,fmt='%.4f')

"""
L = errormap.shape[0]
for i in range(0,L):
   for j in range(i+1,L):
       f.write(str(i+1)+" "+str(j+1)+" "+str(errormap[i][j])+"\n")
f.close()
"""
