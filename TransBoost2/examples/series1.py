# -*- coding: utf-8 -*-
"""
This code applies TransBoost to a source dataset and target train, test and unknown dataset and prints the results.
It requires running series0.py previously.
"""

from __future__ import print_function, absolute_import
import numpy as np

import transBoost.transBoost as transBoost
import transBoost.projections as projections
import tools.data as data
import series.series as series
from series.seriesGeneration import cutSeries

def run():
    #Parameters
    L=150 #length of source series
    l=30 #length of target series
    
    #Data extraction : We need a source dataset of series of length L and a target dataset of series of length l
    X_,y_= data.importCSV("seriesDATASETS/seriesDataset_dev0.02_slp400.txt", delimiter = '\t') #all data
    X_source,y_source,X_train,y_train,X_test,y_test,X,y = data.randomSample(X_[:], y_[:], prop=(0.4,0.2,0.3,0.005)) #Sampling of required datasets
    #Cutting series to their required length
    X_source = cutSeries(X_source,L)
    X_train = cutSeries(X_train,l)
    X_test = cutSeries(X_test,l)
    X = cutSeries(X,l)
    
    #Training of the source hypothesis
    hs=series.svmhyp(X_source, y_source)
    
    #Definition of the TransBoost class
    tb = transBoost.TransBoost() #Create a TransBoost object
    tb.setNumberSteps(10) #set number of boosting steps
    tb.setTrainSet(X_train,y_train) #set train data
    tb.setTestSet(X_test,y_test) #set test data
    
    #Definition of the ProjFinder class           
    pf = projections.ProjFinder(mode="random",timelimit=20) #Create a projFinder object
    #Projection search space : we will add several projection functions
    f = series.polyline #a projection function
    param= series.polyline_param(L,l) #projection various parameters
    pf.addFunction(f, param) #add a projections collection
    f1 = series.sinus
    param1=series.sinus_param(L)
    pf.addFunction
    #Set the source hypothesis
    pf.setSourceHyp(hs)
    #Set the projFinder
    tb.setProjFinder(pf)
    
    #Run the algorithm
    print("Training phase")
    eTrain=tb.learn() #train TransBoost
    print("Training error : %f" % eTrain)
    print("Testing phase")
    _,eTest=tb.test() #test TransBoost
    print("Testing error : %f" % eTest)
    print("\nDetailed logs :")
    tb.printLog()
    print("Dataset : correct labels")
    print(y)
    print("Dataset : labels predicted by TransBoost")
    y_pred=tb.run(X)
    print(y_pred)