#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 01:15:13 2019

@author: farhan
"""
import os
from evaluate_dncon2style_jie import evaluate

#list_file = os.sys.argv[1] # "/storage/hpc/data/fqg7h/DeepCov/data/val_195/val_195ex27.txt
list_file = "/storage/hpc/data/fqg7h/DeepCov/training/Res18/test.lst"
featureDir = "/storage/hpc/data/fqg7h/DeepCov/data/val_195/cov/"
modelfile = "/storage/hpc/data/fqg7h/DeepCov/training/Res18/FINAL_fullmap_metapsicov_model.npz"
trueMapDir ="/storage/hpc/data/fqg7h/DeepCov/data/val_195/map/"
evaDir = "/storage/hpc/data/fqg7h/DeepCov/training/Res18/evalDir/"
os.system("mkdir "+evaDir)

(acc_l5, acc_l2, acc_1l)=evaluate (list_file, featureDir, modelfile, evaDir, trueMapDir)

with open ("result_output.out", "w+") as f:
    f.write("Precc L/5 = "+str(acc_l5)+"\n")
    f.write("Precc L/2 = "+str(acc_l2)+"\n")
    f.write("Precc L = "+str(acc_1l)+"\n")