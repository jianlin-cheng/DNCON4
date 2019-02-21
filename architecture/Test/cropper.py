#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 19:04:18 2018

@author: farhan
"""

import numpy as np
import sys
import os
from math import sqrt
from math import floor



def readSection(filename:str, section:str): #reads the file. Returns section to be read
        
        with open (filename, "r") as f:
            for line in f:
                if (line.strip() == section):
                    
                    return f.readline().strip()
                
        #return ln
######################################################################################################################################
def getWindow(array, initial_x, final_x):
    x = initial_x
    kernel = final_x
    window = np.zeros((kernel-x,kernel-x))
    for x in range (kernel):
          for y in range (x,kernel):
                    window[x,y]=array[x,y]
                    window[y,x]=array[y,x]
    return window
######################################################################################################################################
def getWindowValues(array, rowx, coly, kernel, length):
    x = rowx
    y= coly
    window = np.zeros((kernel,kernel))
    for i in range(x,x+kernel):
        for j in range(y,y+kernel):
            
            if (j<=length-1 and i<=length-1):
                
                window[i-x,j-y]=array[i,j]
            else:
                window[i-x,j-y]=0
    return window

######################################################################################################################################
def getWindow2(array, initial_x, final_x):
    x = initial_x
    kernel = final_x
    window = np.zeros((kernel-x,kernel-x))
    window = array[x:kernel,x:kernel]
    
    return window
######################################################################################################################################
"""def crop(array, kernel,stride,filename,length):
    l=[]
    #count = floor((length-kernel)/stride)
    
    for i in range(0,length,stride):
        #count-=1
        for j in range(0,length,stride):#Needs work
            if (j+kernel>=length):
                l.append(getWindowValues(array,i,j,kernel,length))
                break
            l.append(getWindowValues(array,i,j,kernel,length))
            
        if (i+kernel>=length):
            for j in range(0,length,stride):#Needs work
                if (j+kernel>=length):
                    l.append(getWindowValues(array,i,j,kernel,length))
                    break
                l.append(getWindowValues(array,i,j,kernel,length))
            break
    return l
"""
def crop(array, kernel,stride,filename,length):
    l=[]
    #count = floor((length-kernel)/stride)
    
    for i in range(0,length,stride):
        if (i+kernel>=length):
            
            for j in range(0,length,stride):#Needs work
                if (j+kernel>=length):
                    l.append(getWindowValues(array,i,j,kernel,length))
                    i+=kernel
                    break
                l.append(getWindowValues(array,i,j,kernel,length))
            
            break
        
        for j in range(0,length,stride):#Needs work
            if (j+kernel>=length):
                l.append(getWindowValues(array,i,j,kernel,length))
                
                break
            
            l.append(getWindowValues(array,i,j,kernel,length))
    return l
######################################################################################################################################
def compare(a1:np.ndarray,a2:np.ndarray):
    a1=a1.flatten()
    a2=a2.flatten()
    if len(a1)!=len(a2):
        return False
    
    for i in range(len(a1)):
        if (a1[i]!=a2[i]):
            return False
    return True
######################################################################################################################################
def loadParameters(*file):
    if len(file)==0:
        param="parameter.txt"
    else:
        param= file[0]
    with open (param) as f:
        for line in f:
            if line.startswith("k"):
                kernel = int(line.strip().split("=")[1])
            if line.startswith("s"):
                stride = int(line.strip().split("=")[1])
    return kernel,stride
######################################################################################################################################
def writetoFile(ls, filename, contains):
    ofile = filename.strip().split(".txt")[0]
    ofile+="-"+contains
    path = os.getcwd()+"/crop/"
    
    for i in range(len(ls)):
        #with open (ofile+str(i)+".crop","w+") as f:
        ls[i].tofile(path+ofile+"-"+str(i)+".crop","")
        
    
######################################################################################################################################
def checkValues(file,ls):
    b=np.fromfile(file)

    #b=b.reshape(24,24)
    print("###########################33B")
    print(len(b))
    l = ls[1].flatten()
    for i in range(len(l)):
        print(str(b[i])+"                 "+str(l[i]))
        if (b[i]!=l[i]):
            return False
    return True
######################################################################################################################################
def init(file,sectionlist:list):
    kernel,stride = loadParameters()
    
    for sec in range(len(sectionlist)):
        
        #print(sectionlist[sec])
        
        lnlist=readSection(file,sectionlist[sec]).split()
        
        length = int(sqrt(len(lnlist)))
        
        array = np.zeros((length*length),dtype=np.float32)
        
        for i in range(len(lnlist)):
            array[i]=np.float32(lnlist[i])
        array=array.reshape(length,length)
        ls = crop(array,kernel,stride,"",length)

        writetoFile(ls,file,sectionlist[sec].strip("# "))
    return ls
       
######################################################################################################################################

######################################################################################################################################
sectionlist=["# ccmpred","# pstat_pots","# pstat_mimt","# pstat_mip"]
#sectionlist=["# ccmpred"]
#originalfilename = "X-1A1X-A.txt"
filelist= sys.argv[1]
with open(filelist) as f:
    for file in f:
        init(file.strip(),sectionlist)

#print(ls[0])

"""
lnlist=readSection("X-1A1X-A.txt","# ccmpred").split()
length = int(sqrt(len(lnlist)))
ccmpred = np.zeros((length*length),dtype=np.float32)
#ccmpred [:]= float(lnlist[:])
for i in range(len(lnlist)):
    ccmpred[i]=np.float32(lnlist[i])
ccmpred=ccmpred.reshape(length,length)
ls = crop(ccmpred,kernel,stride,"",length)

writetoFile(ls,originalfilename,"ccmpred")

"""

