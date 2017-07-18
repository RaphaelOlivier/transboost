# -*- coding: utf-8 -*-
"""

@author: Sema Akkoyunlu, Raphaël Olivier

Boosting core algorithm
"""
from __future__ import print_function, absolute_import, division
import numpy as np
import time

from tools.data import *
from tools.learning import *

def boosting(X, y, K, projFinder):
    """ Computes for each boosting step Adaboost weights. The projection reasearch part is encapsulated in ProjFinder.search()

    Parameters
    --------
    X : [n_examples][l1_features] (target examples)
    y : [n_examples] (labels)
    K : int (number of boosting steps)
    projFinder : ProjFinder (projections research class [l1_features] -> [l2_features]). Contains the source hypothesis

    Returns
    -------
    projs : Projections() (class containing projections of type [l1_features] -> [l2_features] that Adaboost retained)
    e : [n_projs] (ordered list of projection errors) n_projs=K or n_projs<K
    a : [n_projs] (ordered list of factors given to each projection)
    t : [n_projs] (exploration time required to find each projection)
    """
    N = X.shape[0] # Number of examples
    D = np.array(np.ones(N)/N) # Weights of each example, initially 1/N
    a = []
    t = []
    e = []
    projFinder.init(X,y) # ProjFinder is initialised with training data and source hypothesis
    for k in range(K): # At each step
        begin=time.time()
        print('Boosting : etape',k)
        y_pred,err = projFinder.search(D) # Looking for a projection (How we look for it depends on ProjFinder)
        end=time.time()
        if(err==None): # If we found none, we stop here
            print("Aucun continuateur performant : interruption du boosting après "+str(k)+" étapes")
            break
        if err==0: # If we found a perfect projector, we keep it and stop here
            projFinder.keepLast()
            a=[1] # One coefficient
            e.append(0) #Null error
            t.append(end-begin) #Exploration time
            break
        alpha = 0.5*np.log((1-err)/err) # Usual case : we compute the coefficient to give to this projection

        D = computeWeights(D, y, y_pred, err) # Update example weights
        
        a.append(alpha)
        t.append(end-begin)
        e.append(err)
        
    projs = projFinder.getProjections() # End of boosting : we get projections in a projection class.
    return projs, e, a, t


def computeWeights(D, y, y_pred, err):
    """ Update weights of examples after each boosting step

    Parameters
    --------
    y [n_examples] (correct labels)
    y_pred [n_examples] (predicted labels)
    D [n_examples] weights assigned to each example

    Returns
    -------
    D [n_examples] updated weights

    """
    rightCoeff=0.5/(1-err) #Factor by which we multiply the weights of correct examples
    wrongCoeff=0.5/err #Factor by which we multiply the weights of wrong examples
    T = y*y_pred #Product of expected and predicted class
    for i in range(len(T)):
        if T[i] == 1: #Prediction is correct
            D[i] = rightCoeff*D[i]
        else: #Prediction is wrong
            D[i] = wrongCoeff*D[i]
    #The sum remains the same
    return D

def test(X, y, projs, alphas):
    """ Testing our projections : For each of them we project the examples and apply the source hypothesis.
    Final projection is given by the sign of the weighted sum of partial predictions.

    Parameters
    -------
    X : [n_examples][l1_features] (target examples)
    y : [n_examples] (labels)
    projs : Projections (class containing n_projs projections of type [l1_features] -> [l2_features]). Also contains the source hypothesis
    alphas : [n_projs]

    Returns
    -------
    err : int
    y_pred : [n_examples]
    """
    y_proj = projs.labelsList(X) #Returns the list of partial predictions for each projection and for the source hypothesis
    errprojs=[]
    y_pred = np.zeros(len(y)) #Initializing the sum
    for i in range(len(alphas)):
        errprojs.append(weightedError(y, y_proj[i])) #Appends the partial error
        y_pred = y_pred+alphas[i]*y_proj[i] #Adding the weighted prediction
    y_pred=np.sign(y_pred) #Final prediction
    err=weightedError(y_pred,y) #Final error
    return err,y_pred, errprojs

def run(X, projs, alphas):
    """ Running our projections on a target dataset : For each of them we project the examples and apply the source hypothesis.
    Final projection is given by the sign of the weighted sum of partial predictions.

    Parameters
    -------
    X : [n_examples][l1_features] (target examples)
    projs : Projections (class containing n_projs projections of type [l1_features] -> [l2_features]). Also contains the source hypothesis
    alphas : [n_projs]

    Returns
    -------
    y_pred : [n_examples]
    """
    y_proj = projs.labelsList(X) #Returns the list of partial predictions for each projection and for the source hypothesis
    y_pred = np.zeros(len(y)) #Initializing the sum
    for i in range(len(alphas)):
        y_pred = y_pred+alphas[i]*y_proj[i] #Adding the weighted prediction
    y_pred=np.sign(y_pred) #Final prediction
    return y_pred

