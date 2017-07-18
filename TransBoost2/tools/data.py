# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:07:37 2015

@author: Sema Akkoyunlu, Raphaël Olivier

Files and datasets manipulation
"""

from __future__ import print_function, absolute_import
import numpy as np
import csv
    
def importCSV(path, delimiter):
    """
    Import a csv file into a dataset (numpy arrays). The csv file must have examples in lines with the first column being labels, the others being features.
    
    Parameters
    ----------
    path : String (relative or absolute path to the existing csv file)
    delimiter : char (delimiter used in the csv)
    
    Returns
    -------
    X : [n_examples][n_features] (points)
    y : [n_examples] (labels)
    """
    
    f = open(path)
    csv_f = csv.reader(f, delimiter = delimiter)
    a = []
    for row in csv_f:
        row = [float(i) for i in row]
        a.append(row)
    N_lines = len(a)
    N_col = len(a[0])
    arr = np.zeros((N_lines, N_col))
    for i in range(N_lines):
        arr[i,:] = a[i]
    X = arr[:, 1:]
    y = arr[:,0]   
        
    return X, y
    
def exportCSV(X,y,chemin,delimiter):
    
    """
    Export a dataset (numpy arrays) into a csv file. The file will have examples in lines with the first column being labels, the others being features.
    
    Parameters
    ----------
    X : [n_examples][n_features] (points)
    y : [n_examples] (labels)
    path : String (relative or absolute path to the existing csv file)
    delimiter : char (delimiter used in the csv)
    """
    
    Z=np.append(y.reshape(X.shape[0],1),X,axis=1)
    np.savetxt(chemin,Z,delimiter=delimiter)
    
def randomSample(X, y,prop=(0.3)):
    """
    Extract one or several random samples in a dataset, without replacement
    
    Parameters
    ----------
    X : [n_examples][n_features] (points)
    y : [n_examples] (labels)
    prop : tuple of m floats (proportion of the whole data to use in each sample)
    
    Returns
    -------
    tuple of 2m arrays : ([n1_examples][n_features],[n1_examples],...,[n2_examples][n_features],[n2_examples]]) (corresponds to (X1,y1,...,Xm,ym))
    """
    
    l=[]
    N=len(y)
    I=np.arange(0, N,1)
    for i in range(len(prop)):
        n = min(int(N*prop[i]),len(I))
        i = np.random.choice(I, size=n, replace=False)
        l.append(X[i])
        l.append(y[i])
        I = np.delete(I, i)
        
    return tuple(l)

def inverseLabels(y): #Inverse all labels
    return -y