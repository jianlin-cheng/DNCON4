#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 03:28:47 2019

@author: farhan
"""
from __future__ import print_function
import sys
project_root = '/data/farhan/SoftwareTools/DeepCov/training/reproduce/scripts/jie_training_code/jie_training_code/ResNet/'
sys.path.insert(0, project_root)

import numpy as np
import os
from math import sqrt
import sys

import time

import theano
import theano.tensor as T

import lasagne
import fnmatch
# This assumes nndef.py is in pwd
from nndef import build_cnn

# Locations of .map and .21c files (latter calculated from .aln files)
#dc_train_map_dir = '/mnt/data/jiehou/DeepCov/reproduce/training/map/' #DeepCov .map Label file path
#dc_train_21c_dir = '/mnt/data/jiehou/DeepCov/reproduce/training/aln21stats/' #DeepCov .21stats binary input files
#dc_test_21c_dir = "/mnt/data/jiehou/Python/DNCON4/data/badri_training_benchmark/feats/" #DNCON2 .21stats Test Data = 195
#dc_label_Y80_dir = "/mnt/data/jiehou/Python/DNCON4/data/badri_training_benchmark/feats/" #DNCON2 .txt Y80- Labels
#pred_dest = "/data/farhan/SoftwareTools/DeepCov/training/reproduce/results/pred_cmap/" #.cmap files kept here after prediction
############################################################################################################################
"""
def deepCovPredictCmap(pdb):
    if (os.path.exists(dc_test_21c_dir+pdb+".21stats")):
        covfile=dc_test_21c_dir+pdb+".21stats"    
    elif (os.path.exists(dc_train_21c_dir+pdb+".21stats")):
        covfile = dc_train_21c_dir+pdb+".21stats"
    else:
        print (pdb+ " not found in known directories...Moving on")
        return
    #covfile=dc_test_21c_dir+pdb+".21stats"
    os.system ("python predictor.py "+covfile)
    cmapfile=pdb+".cmap"
    pred = np.loadtxt(pred_dest+cmapfile,dtype = np.float32)
    return pred
"""
##########################################################################################################################################

# Floor everything below the triangle of interest to zero
def floor_lower_left_to_zero(P, min_seq_sep):
    X = np.copy(P)
    #print("Shape of P or XP is "+str(X.shape))
    datacount = len(X[:, 0])
    #print("Datacount = ",datacount)
    L = int(sqrt(len(X[0, :])))
    print("Length at floor =",L)
    X_reshaped = X.reshape(datacount, L, L)
    for p in range(0,L):
        for q in range(0,L):
            if ( q - p < min_seq_sep):
                X_reshaped[:, p, q] = 0
    X = X_reshaped.reshape(datacount, L * L)
    return X

# Ceil top xL predictions to 1, others to zero

##########################################################################################################################################

def ceil_top_xL_to_one(ref_file_dict, XP, Y, x):
	X_ceiled = np.copy(XP)
	i = -1
	for pdb in sorted(ref_file_dict):
		i = i + 1
		xL = int(x * ref_file_dict[pdb])
		X_ceiled[i, :] = np.zeros(len(XP[i, :]))
		X_ceiled[i, np.argpartition(XP[i, :], -xL)[-xL:]] = 1
	return X_ceiled
##########################################################################################################################################

def print_detailed_evaluations(dict_l, dict_n, dict_e, PL5, PL2, PL, Y, epoch=None, FILE_FLAG=1):
    datacount = len(dict_l)
    epsilon = 1e-07
    if epoch==None:
        epoch = "N/A"
    avg_nc  = 0    # average true Nc
    avg_pc_l5  = 0 # average predicted correct L/5
    avg_pc_l2  = 0 # average predicted correct L/2
    avg_pc_1l  = 0 # average predicted correct 1L
    avg_acc_l5 = 0.0
    avg_acc_l2 = 0.0
    avg_acc_1l = 0.0
    list_acc_l5 = []
    list_acc_l2 = []
    list_acc_1l = []
    i = -1
    if (FILE_FLAG==1):
        if(os.path.exists("Evaluations.log")):
            print("Removing existing Evaluations.log file...")
            os.system("rm -f Evaluation.log")
        with open("Evaluations.log","w+") as f:
            print("New Evaluation.log file created in directory "+os.path.abspath("Evaluation.log"))
            f.write ("  Epoch    PDB       L      Nseq    Neff            Nc      L/5    PcL/5    PcL/2    Pc1L    AccL/5     AccL/2     AccL\n")
            f.write ("-----------------------------------------------------------------------------------------------------------------------\n")
    
    print ("  Epoch    PDB       L      Nseq    Neff            Nc      L/5    PcL/5    PcL/2    Pc1L    AccL/5     AccL/2     AccL")
    print ("-----------------------------------------------------------------------------------------------------------------------")
    
    for pdb in sorted(dict_l):
        i = i + 1
        nc = int(Y[i].sum())
        L = dict_l[pdb]
        L5 = int(L/5)
        L2 = int(L/2)
        pc_l5 = np.logical_and(Y[i], PL5[i, :]).sum()
        pc_l2 = np.logical_and(Y[i], PL2[i, :]).sum()
        pc_1l = np.logical_and(Y[i], PL[i, :]).sum()
        acc_l5 = float(pc_l5) / (float(L5) + epsilon)
        acc_l2 = float(pc_l2) / (float(L2) + epsilon)
        acc_1l = float(pc_1l) / (float(L) + epsilon)
        list_acc_l5.append(acc_l5)
        list_acc_l2.append(acc_l2)
        list_acc_1l.append(acc_1l)
        si = str(i+1)
        spdb = str(pdb)
        sdict_n = str(dict_n[pdb])
        sdict_e = str(dict_e[pdb])
        sL5 = str(L5)
        sL = str(L)
        snc = str(nc)
        spc_l5 = str(pc_l5)
        spc_l2 = str(pc_l2)
        spc_1l = str(pc_1l)
        sacc_l5 = str(round(acc_l5,4))
        sacc_l2 = str(round(acc_l2,4))
        sacc_1l = str(round(acc_1l,4))
        sepoch = str(epoch)
        #print ("  ID    PDB       L      Nseq    Neff            Nc      L/5    PcL/5    PcL/2    Pc1L    AccL/5     AccL/2     AccL")
        #print ("-----------------------------------------------------------------------------------------------------------------------")
        #print ("  "+ sepoch+" "*(12-len(spdb))+ spdb+" "*(7-len(sL))+  sL+" "*(6-len(sdict_n))+ sdict_n+ " "*(14-len(sdict_e))+ sdict_e+" "*(9-len(snc))+ snc+" "*(6-len(sL5))+ sL5+" "*(7-len(spc_l5))+ spc_l5+" "*(9-len(spc_l2))+ spc_l2+" "*(9-len(spc_1l))+ spc_1l+" "*(10-len(sacc_l5))+ sacc_l5+" "*(13-len(sacc_l2))+ sacc_l2+" "*(10-len(sacc_1l))+ sacc_1l)
        
        #print(sdict_e)
        if (FILE_FLAG==1):
            with open("Evaluations.log","a") as f:
                f.write("  "+ sepoch+" "*(11-len(spdb))+ spdb+" "*(7-len(sL))+  sL+" "*(8-len(sdict_n))+ sdict_n+ " "*(15-len(sdict_e))+ sdict_e+" "*(9-len(snc))+ snc+" "*(6-len(sL5))+ sL5+" "*(7-len(spc_l5))+ spc_l5+" "*(9-len(spc_l2))+ spc_l2+" "*(9-len(spc_1l))+ spc_1l+" "*(10-len(sacc_l5))+ sacc_l5+" "*(13-len(sacc_l2))+ sacc_l2+" "*(10-len(sacc_1l))+ sacc_1l+"\n")
                
        avg_nc = avg_nc + nc
        avg_pc_l5 = avg_pc_l5 + pc_l5
        avg_pc_l2 = avg_pc_l2 + pc_l2
        avg_pc_1l = avg_pc_1l + pc_1l
        avg_acc_l5 = avg_acc_l5 + acc_l5
        avg_acc_l2 = avg_acc_l2 + acc_l2
        avg_acc_1l = avg_acc_1l + acc_1l
    avg_nc = int(avg_nc/datacount)
    avg_pc_l5 = int(avg_pc_l5/datacount)
    avg_pc_l2 = int(avg_pc_l2/datacount)
    avg_pc_1l = int(avg_pc_1l/datacount)
    avg_acc_l5 = avg_acc_l5/datacount
    avg_acc_l2 = avg_acc_l2/datacount
    avg_acc_1l = avg_acc_1l/datacount
    
    savg_nc =str(avg_nc)
    savg_pc_l5 = str(avg_pc_l5)
    savg_pc_l2 = str(avg_pc_l2)
    savg_pc_1l = str(avg_pc_1l)
    savg_acc_l5 = str(round(avg_acc_l5,4))
    savg_acc_l2 = str(round(avg_acc_l2,4))
    savg_acc_1l = str(round(avg_acc_1l,4))
    print ("-----------------------------------------------------------------------------------------------------------------------")
    print ("  Avg"+ " "*(48-len(savg_nc))+ savg_nc+ " "*(13-len(savg_pc_l5))+ savg_pc_l5+ " "*(9-len(savg_pc_l2))+ savg_pc_l2+ " "*(9-len(savg_pc_1l))+ savg_pc_1l+ " "*(10-len(savg_acc_l5))+ savg_acc_l5 + " "*(13-len(savg_acc_l2)) + savg_acc_l2 +" "*(10-len(savg_acc_1l))+  savg_acc_1l)
    print ("")
    if (FILE_FLAG==1):
            with open("Evaluations.log","a") as f:
                f.write ("-----------------------------------------------------------------------------------------------------------------------\n")
                f.write ("  Avg"+ " "*(48-len(savg_nc))+ savg_nc+ " "*(13-len(savg_pc_l5))+ savg_pc_l5+ " "*(9-len(savg_pc_l2))+ savg_pc_l2+ " "*(9-len(savg_pc_1l))+ savg_pc_1l+ " "*(10-len(savg_acc_l5))+ savg_acc_l5 + " "*(13-len(savg_acc_l2)) + savg_acc_l2 +" "*(10-len(savg_acc_1l))+  savg_acc_1l+"\n")
    return (list_acc_l5, list_acc_l2, list_acc_1l)
##########################################################################################################################################
##########################################################################################################################################
def evaluate_prediction (maxL,dict_l, dict_n, dict_e, P, Y, min_seq_sep,epoch):
	P2 = floor_lower_left_to_zero(P, min_seq_sep)
	#datacount = len(Y[:, 0])
	#L = int(sqrt(len(Y[0, :])))
	#Y1 = floor_lower_left_to_zero(Y, min_seq_sep)
	list_acc_l5 = []
	list_acc_l2 = []
	list_acc_1l = []
	P3L5 = ceil_top_xL_to_one(dict_l, P2, Y, 0.2)
	P3L2 = ceil_top_xL_to_one(dict_l, P2, Y, 0.5)
	P31L = ceil_top_xL_to_one(dict_l, P2, Y, 1)
	(list_acc_l5, list_acc_l2, list_acc_1l) = print_detailed_evaluations(dict_l, dict_n, dict_e, P3L5, P3L2, P31L, Y, epoch)
	return (list_acc_l5, list_acc_l2, list_acc_1l)
###########################################################################################################################
def getNandNeff(n_file, neff_file, pdb_list):
    all_n = {}
    all_e ={}
    print(os.path.exists(neff_file))
    if (os.path.exists(n_file)):
        with open (n_file,"r") as f:
            for line in f:
                
                pdb = line.strip().split()[0]
                
                value = int(line.strip().split()[1])
                all_n[pdb]=value
    else:
        for pdb in pdb_list:
            all_n[pdb]="N/A"
    
    if (os.path.exists(neff_file)):
        with open (neff_file,"r") as f:
            for line in f:
                pdb = line.strip().split()[0]
                value = float(line.strip().split()[1])
                all_e[pdb]=value
    else:
        for pdb in pdb_list:
            all_e[pdb]="N/A"
   
    # For those proteins not related to DNCON2
    for pdb in pdb_list:
        if not (pdb in all_n.keys()):
            all_n[pdb]="N/A"
        if not (pdb in all_e.keys()):
            all_e[pdb]="N/A"
        
    return all_n,all_e
###########################################################################################################################
def print_detailed_accuracy_on_this_data(pdb_list,predictMapDir,trueMapDir,epoch):
    
    print ('')
    all_list_acc_l5 = []
    all_list_acc_l2 = []
    all_list_acc_1l = []
    n_file = "N.txt"
    neff_file = "Neff.txt"
    all_n,all_e = getNandNeff(n_file,neff_file, pdb_list)
    
    L = []
	#X = []
    Y = []
    P_cmap = [] #Predicted cmap of labels
    dict_l = {}
    deepcov_flag = 0
    for pdb in pdb_list:
        
        if (os.path.exists(trueMapDir+"/Y80-"+pdb+".txt")):
            rawdata = np.loadtxt(trueMapDir+"/Y80-"+pdb+".txt",dtype=np.float32) #for DNCON2 Data
            length = len(rawdata)
            deepcov_flag = 0
        elif (os.path.exists(trueMapDir + '/' + pdb + '.map')):
            data = np.memmap(trueMapDir + '/' + pdb + '.map', dtype=np.float32, mode='r')#DeepCov Data
            length = int(sqrt(data.shape[0]/(61)))# for DeepCov
            deepcov_flag = 1
            rawdata = data.reshape(61, length, length)[0,:,:]
            rawdata.reshape(length,length)
        else:
            print("Labels for ", pdb, "not found...Moving on!")
            continue
                
        
        cmapfile = predictMapDir +"/" +pdb+".cmap"
        if (not os.path.exists(cmapfile)):
            print("Failed to find ",cmapfile, " for evaluation")
            continue
        pred = np.loadtxt(cmapfile,dtype = np.float32)
        
        #rawdata = np.memmap(dc_train_map_dir + pdb + '.map', dtype=np.float32, mode='r')#DeepCov Data
        #rawdata=rawdata.reshape(61, length, length)[0,:,:]
        #Get the array from the Y80-pdb.txt file
        #rawdata = np.loadtxt(dc_label_Y80_dir+"Y80-"+pdb+".txt",dtype=np.float32) #for DNCON2 Data
        #length = len(rawdata)
        #length = int(sqrt(rawdata.shape[0]/(61)))# for DeepCov
        dict_l[pdb]=length
        Y.append(rawdata)
        L.append(length)
        #if 'TRAIN' in id_string:
        #print ('Printing detailed results for TRAIN group ' + str(group))
        #P = make_prediction(model_arch, file_weights, XRT1)
        #print("Predicting contact map using current weights...")
        P_cmap.append(pred)
        #Not Sure if X is needed. Looks similar to pdb_list
    
    maxL = max(L)
    print("Maximum length= ",maxL)
    #print(L[0], L[1])
    
    # adjust the arrays to MaxL
    new_Y = np.zeros((len(Y),maxL,maxL))
    new_P = np.zeros((len(Y),maxL,maxL))
    
    for i in range(len(Y)):
        new_Y[i,0:L[i],0:L[i]]= Y[i] 
        #new_Y[i]=new_Y[i].flatten()
        new_P[i,0:L[i],0:L[i]]= P_cmap[i] 
        #new_P[i]=new_P[i].flatten()
    Y = new_Y
    P_cmap = new_P
    #Y[0]=Y[0].flatten()
    Y = Y.reshape(len(L),maxL*maxL)
    P_cmap = P_cmap.reshape(len(L), maxL * maxL)
    
    #print("Shape pf P =",P_cmap.shape)
    
    (list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(maxL,dict_l,all_n, all_e, P_cmap, Y, 24, epoch)
    all_list_acc_l5.extend(list_acc_l5)
    all_list_acc_l2.extend(list_acc_l2)
    all_list_acc_1l.extend(list_acc_1l)
    acc_l5 = sum(all_list_acc_l5) / len(all_list_acc_l5)
    acc_l2 = sum(all_list_acc_l2) / len(all_list_acc_l2)
    acc_1l = sum(all_list_acc_1l) / len(all_list_acc_1l)
    #print(acc_l5, acc_l2, acc_1l)
    return (acc_l5, acc_l2, acc_1l)
    """
    print ('----------------------------------------------------------------------------------------------')
    print ('Cycle DataSet      Acc-L/5  Acc-L/2  Acc-L')
    print ('' + id_string + ' %.3f    %.3f    %.3f' %(acc_l5, acc_l2, acc_1l))
    print ('----------------------------------------------------------------------------------------------')
    """
"""			
	if 'TEST' in id_string:
	print ('Printing detailed results for TEST group ' + str(group))
		
	#P = make_prediction(model_arch, file_weights, XTE1)
    P_cmap = deepCovPredictCmap(pdb)
	(list_acc_l5, list_acc_l2, list_acc_1l) = evaluate_prediction(LTE1, all_n, all_e, P, YTE1, 24)
	all_list_acc_l5.extend(list_acc_l5)
	all_list_acc_l2.extend(list_acc_l2)
	all_list_acc_1l.extend(list_acc_1l)
"""
def checkFileExists(pdb_list, pathX, pathY): #check if the protein files exist in the mentioned paths
    new_list = []
    print("Checking to see if the files in the list exist...")
    for pdb in pdb_list:
        if (os.path.exists(pathX[0]+pdb+".21stats") or os.path.exists(pathX[1]+pdb+".cov")):
            
            if (os.path.exists(pathY[0]+pdb+".map") or os.path.exists(pathY[1]+"Y80-"+pdb+".txt")):
                new_list.append(pdb)
            else:
                print("Labels for "+pathY[0]+pdb+".21stats or"+pathY[1]+"Y80-"+pdb+".txt not found!...Excluding this from the prediction list!\n")
        else:
            print("The file "+ pathX[0]+pdb+".21stats or"+ pathX[1]+pdb+".cov not found in known directories!... Excluding this from the prediction list!\n")
                
    #print("Predicting for the follwing files: ")
    #for prot in new_list:
    #    print(prot)
    
    return new_list



def predictor(featureFile,modelFile,outputFile):

    input_var = T.ftensor4('inputs')

    # Create neural network model
    network = build_cnn(input_var)
    
    # Load parameters; the file must be in pwd
    #with np.load(original_weights_path +'FINAL_fullmap_metapsicov_model.npz') as f:
    with np.load(modelFile) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(network, param_values)

    # Load input data (.21stats file)
    mapdata = np.fromfile(featureFile, dtype=np.float32)
    
    length = int(sqrt(mapdata.shape[0]/21/21))
    inputs = mapdata.reshape(1,441,length,length)

    predict = lasagne.layers.get_output(network, input_var, deterministic=True)
    predict_fn = theano.function([input_var], predict)

    # Make the predictions
    result = predict_fn(inputs)

    # Write output to stdout; average values for residue pairs i,j and j,i
    for wi in range(0, length-1):
        for wj in range(wi+1, length):
            #print("{0} {1} 0 8 {2:.12g}".format(wi+1, wj+1, 0.5 * (result[0,0,wi,wj] + result[0,0,wj,wi])))
            avg = 0.5 * (result[0,0,wi,wj] + result[0,0,wj,wi])
            result[0,0,wi,wj] = result[0,0,wj,wi] = avg
    result=np.squeeze(result)
    #result = result.round()
    #split = sys.argv[1].strip().split("/")
    #last = split[len(split)-1]
    cmap_file = outputFile
    #cmap_file = last.split(".")[0]+".cmap"
    
    np.savetxt(cmap_file,result)
    #print("Saving the prediction to ",cmap_file)
    #print (os.path.abspath(sys.argv[1].strip(".21stats")+".cmap"))

def evaluate(eval_list,featureDir, modelFile,evaDir,trueMapDir, featurePrefix='.21stats', epoch=None): #can be a list or ".lst" file. Pass epoch during training
    prot_list=[]

    if (".lst" in eval_list):
        if (os.path.exists(eval_list)):
            print("Found the prediction list file: "+ eval_list)
            print("Reading file and creating a list of proteins...")
            
            with open(eval_list,"r") as f:
                for line in f:
                    prot_list.append(line.strip())
            eval_list = prot_list
            

    print("List contains ... "+str(len(eval_list))+" proteins")        
    prot_list = checkFileExists(eval_list, [featureDir, featureDir] , [trueMapDir, trueMapDir])
    #print(os.path.exists(dc_test_21c_dir+eval_list[0]+".21stats"))
    
    if (len(prot_list)==0):
        print("No files to predict and evaluate...Exiting!")
        sys.exit(-1)
    
    
    #### predict the contact map here 
    cmapnum = len(fnmatch.filter(os.listdir(evaDir), '*.cmap'))
    if cmapnum >0:
        os.system ("rm "+evaDir+'/*cmap')
    for pdb in prot_list:
        if (os.path.exists(featureDir+'/'+pdb+featurePrefix)):
            covfile=featureDir+'/'+pdb+featurePrefix    
        else:
            print (pdb+ " not found in known directories ",featureDir,"...Moving on")
            return #continue

        #covfile=dc_test_21c_dir+pdb+".21stats"
        featureFile = covfile
        cmapfile = evaDir + "/" +pdb+".cmap"
        #print(pdb,"...", end="")
        predictor(featureFile,modelFile,cmapfile)
        
        #os.system ("python predictor.py "+covfile)
        #cmapfile=pdb+".cmap"
        #pred = np.loadtxt(cmapfile,dtype = np.float32)


    (acc_l5, acc_l2, acc_1l) = print_detailed_accuracy_on_this_data(prot_list,evaDir,trueMapDir, epoch)
    os.system ("rm "+evaDir+'/*cmap')
    return (acc_l5, acc_l2, acc_1l)

"""
from evaluate_dncon2style_jie import evaluate
#testfile = "1A34-A"
testfile = '/mnt/data/jiehou/DeepCov/reproduce/training/dncon-test_simple.lst'
#featureDir = '/mnt/data/jiehou/DeepCov/reproduce/training/aln21stats/'
featureDir = '/mnt/data/jiehou/Python/DNCON4/data/badri_training_benchmark/feats/'
modelFile = '/mnt/data/jiehou/DeepCov/reproduce/training/train20190119/weights/model_epoch10.npz'
#trueMapDir = '/mnt/data/jiehou/DeepCov/reproduce/training/map/'
trueMapDir = '/mnt/data/jiehou/Python/DNCON4/data/badri_training_benchmark/feats/'
evaDir = '/mnt/data/jiehou/DeepCov/reproduce/training/test/'
evaluate(testfile,featureDir, modelFile,evaDir,trueMapDir, featurePrefix ='.cov') #predict files from list 
"""

