# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:07:37 2015

@author: Sema Akkoyunlu, Raphaël Olivier

Files and datasets manipulation
"""

import random
import numpy as np
import csv
from display import *
import os

def genere_dataset(N, L, sinFreq=np.arange(5, 15, 0.25), sinRange=np.arange(1,10), sinPhase=np.arange(0, 2*np.pi, np.pi/4), maxSlope=1./300, gaussianDeviation=np.linspace(0.01, 0.05, 5)):
    """
    Creates a dataset with N times series of L values each
    And an array with their label: -1 or 1
    
    Example:
    --------
    genere_dataset(3, 10, 1)
    -> (array([[-0.96, -0.94, -1.03, -0.92, -1.04, -0.98, -1.02, -0.99, -0.94,
         -1.04],
        [ 1.06,  1.03,  1.01,  1.04,  1.02,  1.01,  1.06,  0.92,  0.98,
          0.98],
        [ 0.71,  0.7 ,  0.7 ,  0.7 ,  0.72,  0.72,  0.7 ,  0.71,  0.72,
          0.73]]), array([ 1., -1.,  1.]))    
    """
    random.seed()
    X = np.zeros((N, L))  # creates an array of N lines with L colums of zeros
    y = np.zeros(N)       # create an array with N zeros
    t = np.linspace(1, L, L)  # an array with L values [1., 2. , ... L.]
    
    for i in range(N):    # for every time series
        # choose its parameters (controlling the shape)
        freq = random.choice(sinFreq)
        a=random.choice(sinRange)
        # freq is taken randomly inside [250, 275, 280, ..., 825]
        phi = random.choice(sinPhase)
        # initial phase
        b = ((-1)**i*np.random.ranf())/maxSlope
        sigma = np.random.choice(gaussianDeviation)

        l = -1
        if i%2 == 0: #règle de décision de l'attribution de la classe
            l = 1
        # if the trend a is positive, the label is +1
            
        x = a*np.array(np.sin(2 * np.pi * freq * t + phi)
        + (b * t) + np.random.normal(0, sigma, L))/10
        # a sinusoïdal time series with a trend given by a*t and noise 
        X[i] = x
        y[i] = l
    return X, y



#Tronquer la base pour récupérer les T premières valeurs
def cutSeries(data, l):
    """
    Example:
    --------
    >>> db = genere_dataset(3, 10, 1)
    >>> db
    -> (array([[-1.  , -1.03, -1.03, -1.09, -1.  , -0.92, -0.94, -1.03, -1.02,
         -1.07],
        [ 0.93,  1.03,  0.99,  1.01,  1.  ,  1.  ,  1.01,  0.97,  1.01,
          0.99],
        [ 0.72,  0.69,  0.69,  0.66,  0.72,  0.68,  0.7 ,  0.66,  0.74,
          0.75]]), array([ 1., -1.,  1.]))
    >>> db_0 = db[0]
    >>> db_0
    -> array([[-1.  , -1.03, -1.03, -1.09, -1.  , -0.92, -0.94, -1.03, -1.02,
        -1.07],
       [ 0.93,  1.03,  0.99,  1.01,  1.  ,  1.  ,  1.01,  0.97,  1.01,
         0.99],
       [ 0.72,  0.69,  0.69,  0.66,  0.72,  0.68,  0.7 ,  0.66,  0.74,
         0.75]]) 
    >>> tronquer_base(db_0, 3)
    -> array([[-1.  , -1.03, -1.03],
       [ 0.93,  1.03,  0.99],
       [ 0.72,  0.69,  0.69]])
    """
    if l < len(data[0,:]):
        return data[:, 0:l]
    return data

    
def importCSV(chemin, delimiter):
    f = open(chemin)
    csv_f = csv.reader(f, delimiter = delimiter)
    a = []
    for row in csv_f:
        row = [float(i) for i in row]
        a.append(row)
    N_lines = len(a)
    N_col = len(a[0])
    X = np.zeros((N_lines, N_col))
    for i in range(N_lines):
        X[i,:] = a[i]
    X_ = X[:, 1:]
    y_ = X[:,0]
    y_[y_ == 2] = -1     
        
    return X_, y_
    
def exportCSV(X,y,chemin,delimiter):
    
    Z=np.append(y.reshape(X.shape[0],1),X,axis=1)
    np.savetxt(chemin,Z,delimiter=delimiter)
    
def randomSample(X, y,prop=(0.3)):
    somme = np.sum(y)
    l=[]
    N=len(y)
    I=np.arange(0, N,1)
    for i in range(len(prop)):
        n = min(int(N*prop[i]),len(I))
   
        i = random.sample(I, n)
        l.append(X[i])
        l.append(y[i])
        I = np.delete(I, i)
        
    return tuple(l)

def extractResults(logDir):
    boostingnames = {"boosting_K5_PolLinSinMoyPal.log":5,"boosting_K10_PolLinSinMoyPal.log":10,"boosting_K20_PolLinSinMoyPal.log":20,
                     "boosting_K5_LinSinMoy.log":5,"boosting_K10_LinSinMoy.log":10,"boosting_K20_LinSinMoy.log":20,
    "boosting_K15_Lin.log":"Lin","boosting_K15_Moy.log":"Moy","boosting_K15_Sin.log":"Sin"}
    boostingnames = {"boosting_K5_PolLinSinMoyPal.log":5,"boosting_K10_PolLinSinMoyPal.log":10,"boosting_K20_PolLinSinMoyPal.log":20,
                     "boosting_K5_LinSinMoy.log":5,"boosting_K10_LinSinMoy.log":10,"boosting_K20_LinSinMoy.log":20}
    fname="class.log"
    f = open(logDir+"/"+fname, mode='r')
    testsvm=None
    begtestsvm="testing error pour SVM sur la série tronquée : "
    trainsvm=None
    begtrainsvm="training error pour SVM sur la série tronquée :" 
    testsvmfull=None
    begtestsvmfull="testing error pour SVM sur la série totale : "
    trainsvmfull=None
    begtrainsvmfull="training error pour SVM sur la série totale : "
    trainboosting={5:None,10:None,20:None,"Lin":None,"Sin":None,"Moy":None}
    begtrainboosting="training error sur la série complétée par boosting : "
    testboosting={5:None,10:None,20:None,"Lin":None,"Sin":None,"Moy":None}
    begtestboosting="testing error sur la série complétée par boosting : "
    for s in f.readlines():
        if(s.startswith(begtrainsvmfull)):
            trainsvmfull=float(s[len(begtrainsvmfull):])
        if(s.startswith(begtestsvm)):
            testsvm=float(s[len(begtestsvm):])
        if(s.startswith(begtrainsvm)):
            trainsvm=float(s[len(begtrainsvm):])
        if(s.startswith(begtestsvmfull)):
            testsvmfull=float(s[len(begtestsvmfull):])
        if(s.startswith(begtrainboosting)):
            trainboosting=float(s[len(begtrainboosting):])
        if(s.startswith(begtestboosting)):
            testboosting=float(s[len(begtestboosting):])
    f.close()
    for fname in boostingnames:
        try:
            f = open(logDir+"/"+fname, mode='r')
            for s in f.readlines():
                if(s.startswith(begtrainboosting)):
                    trainboosting[boostingnames[fname]]=float(s[len(begtrainboosting):])
                if(s.startswith(begtestboosting)):
                    testboosting[boostingnames[fname]]=float(s[len(begtestboosting):])
        except:
            pass
 
    return trainsvmfull, testsvmfull, trainsvm, testsvm, trainboosting, testboosting


def inverseLabels(y):
    return -y