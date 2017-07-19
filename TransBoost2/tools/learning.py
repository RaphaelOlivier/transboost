# -*- coding: utf-8 -*-
"""
Various useful, general learning functions
"""
from sklearn import svm
import numpy as np

def weightedError(y, y_pred, D=None):
    """ Error rate is the weighted sum of wrongly predicted examples
    
    Parameters
    -------
    y : [n_examples] (correct labels)
    y_pred : [n_examples] (predcted labels)
    D : [n_exemples] (weights given to each example)
    
    Returns 
    -------
    err : float (classification error)
    """
    if D is None: #If no weights we just use uniform distribution
        N = len(y)
        D = np.array(np.ones(N)/N)
    err = .0
    p = y * y_pred #Product of expected and predicted classes
    for i in range(len(p)):
        if p[i] == -1.: #Negative product means wrong class
            err += D[i]
    return err

def error(y, y_pred):
    """Error rate is the proportion  of wrongly predicted examples
    
    Parameters
    -------
    y : [n_examples] (correct labels)
    y_pred : [n_examples] (predcted labels)
    
    Returns 
    -------
    err : float (classification error)
    """
    
    N = len(y_pred)
    err=.0
    p = y * y_pred #Product of expected and predicted classes
    for i in range(len(p)):
        if p[i] == -1.: #Negative product means wrong class
            err+=1
    err/=N
    return err

def testhyp(h,X,y,D=None):
    """
    Apply the given hypothesis and computes the weighted error.
    
    Parameters
    -------
    h : [n_examples][l_features] -> [n_examples]
    X : [n_examples][l_features] (points to classify)
    y : [n_examples] (correct labels)
    D : [n_examples] (weights given to each example)
    
    Returns 
    -------
    yp : [n_examples] (prediction)
    err : float (classification error)    
    """
    yp=h(X) #Prediction
    err=weightedError(y,yp,D) #Error
    return yp,err
    

def learnSVM(X, y, gamma,):
    """SVM learning with gaussian kernel
    
    Parameters
    ---------
    X : array [n_examples][p_features] (points to classify)
    y : array [n_examples] (labels)
    
    Returns 
    ---------
    err : float (prediction error)
    clf : trained SVM classifier
    
    """
    clf = svm.SVC(kernel = 'rbf', C = 2, gamma = gamma) #define classifier
    clf.fit(X, y) #train classifier
    y_pred = clf.predict(X) #apply classifier on training data
    err = error(y, y_pred) #compute training error
    return err, clf

def testSVM(clf, X, y, D = None):
    """ Tests the classifier's prediction over given data, returns weighted error
    
    Parameters
    ---------
    X : [n_examples][n_features] (points to classify)
    y : [n_examples] (correct labels)
    D = None | [n_examples]
    
    Returns
    --------
    err_test : float (prediction error)
    y_predit : [n_examples] (prediction)
    """
    
    y_pred = clf.predict(X) #apply classifier on data
    err_test = weightedError(y, y_pred, D) #compute test error
    return err_test, y_pred