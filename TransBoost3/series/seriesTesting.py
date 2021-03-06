# -*- coding: utf-8 -*-
"""
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
    """
    Tests TransBoost on given series data with given parameters.
    
    Parameters
    ----------
    L : Integer (length of source series)
    l : Integer (length of target series)
    K : Integer (number of boosting steps)
    hs : [n_examples][L] -> [n_examples] (source hypothesis)
    X_train : [n_train][l] (training target set)
    y_train : [n_train] (training target labels)
    X_test : [n_test][l] (testing target set)
    y_test : [n_test] (testing target labels)
    
    Returns
    -------
    eTrain : float (TransBoost training error rate)
    eTest : float (TransBoost testing error rate)
    """
    
    tb = transBoost.TransBoost() #Create a TransBoost object
    tb.setNumberSteps(K) #set number of boosting steps
    tb.setTrainSet(X_train,y_train) #set train data
    tb.setTestSet(X_test,y_test) #set test data
    
    pf = projections.ProjFinder(mode="random",timelimit=20) #Create a projFinder object
    f = series.polyline #a projection function
    param= series.polyline_param(L,l) #projection various parameters
    pf.addFunction(f, param) #add a projections collection
    pf.setSourceHyp(hs) # set the source hypothesis
    
    tb.setProjFinder(pf) #set the projFinder
    
    eTrain=tb.learn() #train TransBoost
    _,eTest=tb.test() #test TransBoost
    
    log=tb.getLog()
    print(log) #print all informations about the experience in the shell
    
    return eTrain, eTest
    
def svmreg(L, l, hs, X_train, y_train, X_test, y_test):
    """
    An hypothesis based on naive regression. Can be used as an alternate reference hypothesis in the testseries function.
    
    Parameters
    ----------
    L : Integer (length of source series)
    l : Integer (length of target series)
    hs : [n_examples][L] -> [n_examples] (source hypothesis)
    X_train : [n_train][l] (training target set)
    y_train : [n_train] (training target labels)
    X_test : [n_test][l] (testing target set)
    y_test : [n_test] (testing target labels)
    
    Returns
    -------
    eTrain : float (Regression training error rate)
    eTest : float (Regression testing error rate)
    """
    
    def hreg(X):
        Z=np.zeros((X.shape[0],L)) #Will hold the dataset of naively projected series
        for i in np.arange(X.shape[0]):
            x = X[i,:] #for the example i
            l=len(x)
            t = np.arange(L).reshape((L,1)) #times
            clfr = svm.SVR(kernel = 'linear', gamma = 0.03) #SVR regression
            clfr.fit(t[:l], x) #computation of the regression for example i
            x_proj = np.append(x,clfr.predict(t[l:])) #projected series : initial series continuated with the regressed values
            Z[i,:] = x_proj #added the projected series to the dataset
        return hs(Z) #apply the source hypothesis to the projected dataset
    
    y,eTrain = testhyp(hreg,X_train,y_train) #Compute error on the training set (although no actual training)
    y,eTest = testhyp(hreg,X_test,y_test) #Compute error on the testing set
    return eTrain, eTest

def simplifyCSV(path):
    """
    Updates a csv file to keep only 2 significant figures after comma.
    
    Parameters
    ----------
    path : String (path to the csv file)
    """
    df = pd.read_csv(path)
    def simp(x): #auxiliary function that keeps only 2 significant figures (s.f) in a float number, and doesn't change any other type
        if(type(x)==float):
            y=0.01*int(100*x) #2  at most
            if(y>0.1): #if exactly two s.f. we keep this value
                return y
            else: #if only one s.f.
                return 0.001*int(1000*x) #we add one
        return x
    
    df=df.applymap(simp) #apply the function to the whole dataFrame
    df.to_csv(path[:-4]+"_simplified.csv",index=False) #export again
    
def displayExperience(df,x,y):
    """
    Plots a graph based on two column in a dataset. Used for example to plot a point for each experience, with x as the TransBoost error and y as the reference error.
    
    Parameters
    ----------
    df : pd.DataFrame
    x : String (name of the column used as x values)
    y : String (name of the column used as y values)
    """
    l1=df[x].tolist()
    l2=df[y].tolist()
    plt.plot([0.0,0.6],[0.0,0.6],'-',color="black") #line 'y=x'
    plt.plot(l1,l2,'.',color="blue") #points (x[i],y[i]) for every index i
    plt.xlim((0.0,0.6)) #Error rates are usually below 0.5 so a (0,0.6) scale is enough
    plt.ylim((0.0,0.6))
    plt.show() #Show the graph
    
    