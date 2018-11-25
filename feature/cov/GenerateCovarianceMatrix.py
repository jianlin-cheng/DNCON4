#"""
#Created on Thu Nov 15 13:20:19 2018

#@author: Farhan Quadir
#"""
#takes in two command line argument variables
#argv[0]= input alignment file (*.aln)
#argv[1]= output covariant matrix file (*.cov)
#include complete path of .aln file if not present in current directory

import sys
import os


from math import sqrt

import numpy as np

def checkFileExist(i): #method to check if file exists in the current working directory
    if (os.path.isfile(i)):
        return True
    else:
        #print("File "+i+ " does not exists!!!!")
        return False

##########################################################################################
def makeCovMatrix(filename): #Writes a 21 x 21 Covariance Matrix to the File filename
    bindata = np.fromfile("output.bin", dtype=np.float32) #reads binary file output.bin as float type  numbers
    
    length = int(sqrt(bindata.shape[0]/21/21)) #calculates the length of the protein 
    inputs = bindata.reshape(1,441,length,length) #get the entire L x 441 matrix

    allin1 = np.zeros((441)) #vector that stores the 441 covariance values
    for i in range (441):
        allin1[i] = inputs[0][i][0][0]

    allin2D=allin1.reshape(21,21) #reshape the 441 vector into 21 x 21 matrix

    with open (filename,"w") as f: #write the 21 x 21 matrix in allin2D into a text file.
        f.write("#Covariance Matrix\n")
        for i in range (21):
            for j in range (21):
                f.write (str(allin2D[i][j])+" ")
            f.write("\n")
########################################################################################3

ifile=sys.argv[1] #read the first input file name-- has to be .aln file
if (ifile.strip().strip(".aln")==ifile.strip()): #check if extension of input file is .aln.
	sys.exit("Input File "+ifile+" extension is not .aln...Moving on!")

obinfile="output.bin" #intermediary binary file

if (len(sys.argv)>2): #this condition checks if a desired output file is mentioned
    ofile=sys.argv[2] #if desired output filename is given it assigns the output file
    ofile=ofile.strip(".cov")+".cov" #ensures the output extension is always .cov
else:
    ofile=ifile.strip().strip(".aln")+".cov" #if output filename not mentioned generate output file with same name as the input alignment file just change the extension

if (checkFileExist(ofile)): #check if the covariance matrix file already exists. Then skip. No need to do this.
    sys.exit("Covariance Matrix File "+ofile+" already exists...Not doing anything...Moving on!")

if (checkFileExist(ifile)): #Now check if the alignment file is found in the current working directory
    print("Alignment File "+ifile+" found...Great!")
else:
    sys.exit("FAILURE: Alignment File "+ifile+" NOT Found...Moving on!")
    
if (checkFileExist("output.bin")): #Check to see if the intermediary output.bin file is in the current working directory. If so we remove it.
    print("output.bin binary file already exists... So what can we do?")
    print("Overwriting the output.bin binary file...")
    os.system("rm output.bin")
else:
    print("Generating output.bin binary file...")

commandlines =[] #list of linux system commands 
commandlines.append("./MakeCovarianceBinaryFile "+ifile+" "+obinfile) #This uses the DeepCov tool cov21stats.C renamed to MakeCovarianceBinaryFile to generate the output.bin file.

commandlines.append("rm -r output.bin") #We don't need the output.bin file.

os.system(commandlines[0])
print("SUCCESS: Binary File output.bin successfully created...")
print("Creating Covariance Matrix File...")
makeCovMatrix(ofile)
print("SUCCESS: Covariance Matrix File "+ofile+" successfully created...")
print("Removing temporary Binary File output.bin ... It's too big!")
os.system(commandlines[1])
print("SUCCESS: Temporary Binary File output.bin successfully removed...")
