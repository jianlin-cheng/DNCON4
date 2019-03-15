#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:07:39 2019

@author: farhan
"""


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plotHeat(data, vmin= None, vmax = None):
    df = pd.DataFrame(data)
    #df = sns.load_dataset(data)
    #sns.heatmap(df, vmin = 0, vmax = 100, cmap='viridis')
    #sns.heatmap(df, vmin = 0, vmax = 50, cmap='viridis')
    sns.heatmap(df, vmin, vmax, cmap='viridis')
    plt.figure()
    
def savePlot(data, filename, vmin, vmax):
    df = pd.DataFrame(data)
    sns.heatmap(df, vmin, vmax, cmap='viridis')
    #plt.figure()
    plt.savefig(filename)
    plt.show()
    plt.close()

def plotTwo(data1, data2, vmin= None, vmax = None):
    fig, ax =plt.subplots(1,2)
    #sns.heatmap(data1, vmin, vmax, ax=ax[0])
    #sns.heatmap(data2, vmin, vmax, ax=ax[1])
    
    sns.heatmap(data1, vmin, vmax, ax=ax[0], cmap ="viridis")
    sns.heatmap(data2, vmin, vmax, ax=ax[1], cmap = "viridis")
    
    plt.show()
    
def saveTwo(data1, data2, vmin, vmax, filename):
    fig, ax =plt.subplots(1,2)
    #sns.heatmap(data1, vmin, vmax, ax=ax[0])
    #sns.heatmap(data2, vmin, vmax, ax=ax[1])
    
    sns.heatmap(data1, vmin, vmax, ax=ax[0], cmap ="viridis")
    sns.heatmap(data2, vmin, vmax, ax=ax[1], cmap = "viridis")
    
    #plt.show()
    plt.savefig(filename)