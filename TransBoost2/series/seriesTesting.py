# -*- coding: utf-8 -*-
"""
Created in may 2017 

@author: Raphaël Olivier

Functions designed to test the TransBoost algorithm on time series
"""
from __future__ import print_function, absolute_import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import svm

import tools.data as data
from tools.learning import testhyp
import transBoost.transBoost as transBoost
import transBoost.projections as projections
import series.series as series
from series.seriesGeneration import cutSeries

def testseries(L_list, l_list, K_list, ds_list, resultsFile=None,n=10):
    """
    This function is designed to apply transboost on several datasets of time series, with different parameters.
    For each set of parameters and each dataset, n identical experiences are run on different samples of the data.
    Identical experiences will be aggregated, and results are gradually saved in a csv file.
    
    Parameters
    ----------
    
    L_list : [nL] (lengths of source series)
    l_list : [nl] (lengths of target series)
    K_list : [nK] (numbers of boosting steps)
    ds_list : [nL] (datasets to use)
    resultsFile : String (the path of the file in which save the results)
    n : Integer (number of identical experiences to run for each set of parameters and each dataset)
    
    Returns
    -------
    aggres : pd.DatFrame (results)
    
    """
    
    #Columns of the dataframe for a sequence of identical experiences
    cols=["full length", "cut length", "boosting steps", "dataset", "reference train score","reference test score", "transBoost train score", "transBoost test score"]
    
    #Columns of the overall results dataset
    aggcols = ["full length", "cut length", "boosting steps", "dataset", "reference train score","reference train deviation", "reference test score", "reference test deviation", "transBoost train score", "transBoost train deviation", "transBoost test score", "transBoost test deviation"]
    aggres=None #aggregate results
    results=[] #list that firstly contains the results
    
    for L in L_list:
        for l in l_list:
            for K in K_list:
                for ds in ds_list:
                    #Here we wil run n experiences
                    r = [] #list that firstly contains the results of the n experiences
                    
                    X,y= data.importCSV(ds, delimiter = '\t') #import the data
                    X = cutSeries(X,L) #cut to length L
                    
                    X_source, y_source, X, y = data.randomSample(X[:], y[:], prop=(0.5,0.5)) #Separate the set of source data and the set where we sample the target data
                    hs=series.svmhyp(X_source, y_source) #source hypothesis
                    
                    for i in range(n):
                        #Here we run one experience
                        X_train, y_train, X_test, y_test = data.randomSample(X[:], y[:], prop=(0.2,0.6)) #Sampling
                        X_train = cutSeries(X_train,l) #Training dataset
                        X_test = cutSeries(X_test,l) #Test dataset
                        href=series.svmhyp(X_train, y_train) #a naive target hypothesis to use as a performance reference in the results
                        _,eTrainRef=testhyp(href,X_train,y_train) #training error of the naive hypothesis
                        _,eTestRef=testhyp(href,X_test,y_test) #testing error of the naive hypothesis
                        
                        eTrain, eTest= singletest(L, l, K, hs, X_train, y_train, X_test, y_test) #Training and testing error of transBoost (see singletest)
                        t = (L, l, K, ds, eTrainRef, eTestRef, eTrain, eTest) #tuple of results for this experience
                        r.append(t) #added to the list of results
                    res=pd.DataFrame(data=r, columns=cols) #dataFrame of results for n experiences
                    
                    m=res.mean() #means of all values in the n experiences
                    s=res.std() #standard deviation of all values in the n experiences
                    eTrainRef=m["reference train score"]
                    eTrainRefstd=s["reference train score"]
                    eTestRef=m["reference test score"]
                    eTestRefstd=s["reference test score"]
                    eTrain=m["transBoost train score"]
                    eTrainstd=s["transBoost train score"]
                    eTest=m["transBoost test score"]
                    eTeststd=s["transBoost test score"]
                    
                    t=(L, l, K, ds, eTrainRef, eTrainRefstd, eTestRef, eTestRefstd, eTrain, eTrainstd, eTest, eTeststd) #tuple of aggregate values for the n experiences
                    results.append(t) #added to the list of results
                    aggres = pd.DataFrame(data=results, columns=aggcols) #after one set of experience, we compute the overall dataFrame and save partial results
                    if(resultsFile!=None):
                        aggres.to_csv(resultsFile) #Results are saved                   
                    
                    print("End of this experience")
    
    print("End of experiences")
    return aggres
                    
def singletest(L, l, K, hs, X_train, y_train, X_test, y_test):

    X_train = cutSeries(X_train,l)
    X_test = cutSeries(X_test,l)
    
    tb = transBoost.TransBoost()
    
    tb.setNumberSteps(K)
    
    pf = projections.ProjFinder(mode="random",timelimit=20)
    
    f = series.polyline
    param= series.polyline_param(L,l)
#    f=series.line
#    param=series.line_param(L,l)
     
    pf.addFunction(f, param)
    tb.setProjFinder(pf)
    
    
    pf.setSourceHyp(hs)
    
    tb.setTrainSet(X_train,y_train)
    tb.setTestSet(X_test,y_test)
    
    eTrain=tb.learn()
    _,eTest=tb.test()
    log=tb.getLog()
    print(log)
    return eTrain, eTest
    
def svmreg(L, l, hs, X_train, y_train, X_test, y_test):
    print("regression")
    def hreg(X):
        Z=np.zeros((X.shape[0],L))
        for i in np.arange(X.shape[0]):
            x = X[i,:] 
            l=len(x)
            t = np.arange(L).reshape((L,1))
            clfr = svm.SVR(kernel = 'linear', gamma = 0.03)
            clfr.fit(t[:l], x)
            Z[i,:] = np.append(x,clfr.predict(t[l:]))
        return hs(Z)
    y,eTrain = testhyp(hreg,X_train,y_train)
    y,eTest = testhyp(hreg,X_test,y_test)
    return eTrain, eTest

def simplifyCSV(path):
    
    df = pd.read_csv(path)

    def simp(x):
        if(type(x)==float):
            y=0.01*int(100*x)
            if(y>0.1):
                return y
            else:
                return 0.001*int(1000*x)
        return x
    df=df.applymap(simp)
    df.to_csv(path[:-4]+"_simplified.csv",index=False)
    
def displayExperience(df,x,y):
    l1=df[x].tolist()
    l2=df[y].tolist()
    plt.plot([0.0,0.6],[0.0,0.6],'-',color="black")
    plt.plot(l1,l2,'.',color="blue")
    plt.xlim((0.0,0.6))
    plt.ylim((0.0,0.6))
    plt.show()