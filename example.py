# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:24:11 2016

@author: raphaelolivier

"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import testseries
"""

from tests import *
import time
import numpy as np
import transBoost
import projections
import series
import data
#Paramètres généraux
L=200
l_list=5
#b=90
#c=2
K=15

X, y = data.importCSV("../new_TransBoost_sequences/DATASET/capDataset_dev0.02_slp200.txt", delimiter = '\t')
X_hyp, y_hyp, X, y = data.randomSample(X[:], y[:], prop=(0.5,0.5))

hs=series.svmhyp(X_hyp, y_hyp, b, c)
X_train = data.cutSeries(X_train,l)
X_test = data.cutSeries(X_test,l)

tb = transBoost.TransBoost()

tb.setNumberSteps(K)

pf = projections.ProjFinder(mode="random",timelimit=30)
f = series.polyline
param= series.polyline_param(L,l)
pf.addFunction(f, param)
tb.setProjFinder(pf)


tb.setSourceHyp(hs)
for i in range(20):
    X_train, y_train, X_test, y_test = data.randomSample(X[:], y[:], prop=(0.2,0.2))
    X_train = data.cutSeries(X_train,l)
    X_test = data.cutSeries(X_test,l)
    tb.setTrainSet(X_train,y_train)
    tb.setTestSet(X_test,y_test)
    tb.learn()
    tb.run()
    tb.printState()
    
tb.printLog(logFile="log/test.log")
"""
"""
testseries.simplifyCSV("log/test_principal.csv")
"""
df = pd.read_csv("log/test_avec_hypothese_quelconque.csv")

testseries.displayExperience(df,"transBoost train score", "reference train score")
testseries.displayExperience(df,"transBoost test score", "reference test score")