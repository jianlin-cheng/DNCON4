#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:22:30 2019

@author: farhan
"""

import Mantel
from scipy import spatial, stats
import numpy as np
import os, sys

def setup(n):
    
    a=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            a[i][j] = i
            a[j][i] = i
    np.fill_diagonal(a,0)
    return a

def setupRandom(n):
    
    #a=np.zeros((n,n))
    a = np.random.rand(n,n)
    for i in range(n):
        for j in range(i,n):
            a[i][j] = a[j][i]
    np.fill_diagonal(a,0)
    return a

#perform Mantels test
def evaluate(predicted, actual, typ): #typ = "spearman" or "pearson"
    r, p, z = Mantel.test(predicted, actual, 10000, typ, "two-tail")
    return r, p, z

#Rewrite this to choice
def evaluate_distance(pdb_list:list):
    return

def difference(predicted, actual): #calculates the difference between the two matrices
    diff = abs(actual - predicted)
    return diff

#Squares the values
def sq(m):
    return m ** 2

#Averages numpy array
def avg(m):
    return np.average(m)


def getTPFP(diff, l = 0):
    """
    if l == 0 :
        length = diff.shape[0]
    else:
        length = l
    total_TP = 0
    total_FP = 0
    for i in range(length):
        for j in range(i+1, length):
            if (diff[i][j] <1): # count the differences close to zero as TP rest are FP
                total_TP +=1
            else:
                total_FP += 1
    
            
    return total_TP, total_FP
    """
    return

def sigmoid(x, Matrix=False, derive=False):
    if derive:
        return x * (1 - x)
    if Matrix:
        n = x.shape[0]
        m = x.shape[1]
        for i in range(n):
            for j in range(m):
                x[i][j] = 1/(1+np.exp(-x[i][j]))
    return 1 / (1 + np.exp(-x))

#Creates a siqgmoid matrix to imitate a contact map. Saves it to be used to calculate Preccision
def saveSigm(pdb, predicted, actual, saveDir, epoch=None):
    if not (os.path.exists(saveDir)):
        os.system("mkdir "+saveDir)
    length = actual.shape[0]
    eight = np.ones((length,length)) * 8
    diff = eight - predicted #negatives are the insignificant values. Sigmoid will lower there values
    sigm = sigmoid(diff, True, False)
    if epoch != None:
        ep = str(epoch)
    else:
        ep = "NA"
    if (os.path.exists(saveDir+"/"+pdb+"-pred-"+str(ep)+".cmap")):
        os.system("rm -f " + saveDir+pdb+"-pred-"+str(ep)+".cmap")
    np.savetxt(saveDir+"/"+pdb+"-pred-"+str(ep)+".cmap",sigm)
    #print(saveDir+"/"+pdb+"-pred-"+str(ep)+".cmap")
    #print(os.path.exists(saveDir+"/"+pdb+"-pred-"+str(ep)+".cmap"))
    return

#Compares the saved contact map like values to actual contact maps to find precision
def getPreccAll(eval_list, saveDir, trueMapDir, epoch):
     from evaluate_dncon2style_farhan import evaluatePrecc
     return evaluatePrecc(eval_list, saveDir, trueMapDir, epoch)

def getPrecc(predicted, actual, prefix = "L"):
    """
    abs_diff = difference(predicted, actual)
    
    if prefix == "L" or prefix == "L/1" or prefix == "1L":
        length = predicted.shape[0]
    
    TP, FP = getTPFP(abs_diff, length)
    print("TP=",TP,"FP=",FP)
    return TP/(TP+FP)
    """
    return

#performs MSE
def getMse(predicted, actual):
    n = actual.shape[0]
    abs_diff = difference(predicted,actual)
    sq_matrix = sq(abs_diff)
    sumtotal = np.sum(sq_matrix)
    #return sumtotal/(n*(n-1))
    return sumtotal/(n*n)
#Creates a cmap form distance matrix
def makeCMAP(dist, threshold):
    length = dist.shape[0]
    cmap = np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if dist[i][j] < threshold:
                cmap[i][j] = 1
    return cmap

#Calculates Euclidean Distance
def getED(predicted, actual): #Euclinean distance
    length = predicted.shape[0]
    val = getMse(predicted, actual) *length*length #MSE squared
    return np.sqrt(val)

#Calculates mean absolute error
def getMAE(predicted, actual):
    diff = difference(predicted, actual)
    sum_diff = diff.sum()
    return sum_diff/ diff.size

