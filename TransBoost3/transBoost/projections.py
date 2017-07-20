# -*- coding: utf-8 -*-
"""
Projection research and application
"""
from __future__ import print_function, absolute_import
import numpy as np
import time
import random

from tools.learning import testhyp

class ProjFinder:
    """
    Encapsulate projection exploration, given a research mode parameters, training data and source hypothesis.
    Often, a projection is a function and its parameters. However there are changes depending on the research mode used.
    """
    def __init__(self,mode="stupid", randombar=0.4999, threshold=0.45,timelimit=None):
        """Constructor
        
        Parameters
        -------
        mode : String (research mode - see the search function)
        randombar : float (error below it means a projection is better than random choice)
        threshold : float (With error below it we can stop the research instantly. We usually don't require that a selected projection is better ; 
                           but if we found one better, it is good enough.)
        timelimit : float (Time limit for a single projection exploration - use depends on the mode)
        """
        self.mode=mode
        self.projfunctions={} #Dictionnary containing projection types as keys and parameters as values - used except for neural search
        self.randombar=randombar
        self.threshold=threshold
        self.X=None #Training examples
        self.y=None #Training labels
        self.hs=None #Source hypothesis
        self.projections=Projections() #Class used to contain selected projections
        self.lastscore=None #Last error found (if zero, we should stop the boosting)
        self.timelimit=timelimit 
        self.graph=None
        
    def setSourceHyp(self,hs):
        """
        Sets source hypothesis. Used in any search except neural search (which does not exist in python 2.7)
        
        Parameters
        ----------
        hs : [n_examples][l_features] -> [n_examples]
        """
        self.hs=hs
    
    
    def addFunction(self,func, params):
        """
        Add a couple (function, set of parameters) in the exploration space.
        Used in stupidsearch and randomsearch.
        
        Parameters
        ---------
        func : [n_features1],[n_params] -> [n_features2]
        params [n_parameters_sets][n_params]
        """
        self.projfunctions[func]=params
    
    def init(self,X,y,mode=None):
        """
        Sets training data, initializes projections to empty space
        
        Parameters
        ----------
        X : [n_examples][n_features1]
        y : [n_examples]
        
        """
        if(mode!=None):
            self.mode=mode
        self.X=X
        self.y=y
        self.lastscore=None
        
        if(self.mode=="stupid" or self.mode=="random"):
            self.projections=Projections(source_hypothesis=self.hs)
        else :
            if(self.mode=="neural"):
                print("Neural search is not allowed in python 2")
                raise
            else:
                print("Unknown search mode :"+str(self.mode))
                raise
        
    def search(self,D):
        """
        Projection research. Distributes research to a precise function according to the mode
        
        Parameters
        ----------
        D : [n_examples] (weights of the examples)
        
        Returns
        --------
        y_pred : [n_examples] (predicted labels related to the selected projection)
        err : float (error rate of the selected projection)
        """
        if(self.lastscore!=None and self.lastscore>=self.randombar): #If we did not find better than random last time, it is useless to keep searching.
            
            return None,None
        if(self.mode=="stupid"):
            return self.stupidsearch(D)
        if(self.mode=="random"):
            return self.randomsearch(D)
        
        if(self.mode=="neural"):
            return self.neuralsearch(D)
        
    def keepLast(self): #Keeps only the last projection found. Useful if this projection has a null error
        self.projections.keepLast()
        
    def stupidsearch(self,D):
        """
        Exhaustive projection search. Explores the ordered set of projections until one has weighted error lower than self.threshold.
        
        Parameters
        ----------
        D : [n_examples] (Weights of the examples)
        
        Returns
        --------
        y_pred_min : [n_examples] predicted labels related to the selected projection
        err_min : float (error rate of the selected projection)
        """
        err_min=2 #Initializaion
        param_min=None
        p_min=None
        y_pred_min=None
        for p in self.projfunctions: #For each function
            for param in self.projfunctions[p]: #For each parameters set
                X_p = self.projections.proj(self.X, p, param) #Apply the projection
                y_pred, err = testhyp(self.hs, X_p, self.y, D) #Apply the source hypothesis
                if err < err_min: #If bette than previously, this error rate becomes the best rate and e keep the related parameters in memory
                    err_min = err
                    param_min = param
                    y_pred_min = y_pred
                    p_min = p
                if err_min < self.threshold: #If we did better than self.threshold, we stop searching parameters
                    break
            if err_min < self.threshold: #We also stop searching other functions
                break
        self.projections.add(p_min,param_min) #We add the selected projection to the projection set
        self.lastscore=err_min #Update the last score
        return y_pred_min, err_min
    
    def randomsearch(self,D):
        """
        Random projection search. We randomly explore the projection set until one gets better than self.threshold, or there is no more, or (in the search of parameters related to one function) we passed the time limit.
        Random search allow to improve the best error rate faster, with the drawback of non-exhaustive search and possibly missing some good projections. However with self.timelimit high enough the difference between exhaustive search results and random search results is likely to be very small.
        
        Parameters
        ----------
        D : [n_examples] (Weights of the examples)
        
        Returns
        --------
        y_pred_min : [n_examples] predicted labels related to the selected projection
        err_min : float (error rate of the selected projection)
        """
        
        err_min=2 #Initializaion
        param_min=None
        p_min=None
        y_pred_min=None
        for p in self.projfunctions: #For each function
            begin = time.time() #Time at the beginning of the search
            n = len(self.projfunctions[p]) #number of parameters sets
            inds = np.arange(n) #indices of parameters sets to explore
            mask = np.ones(n,dtype=bool) #Will be used not to select several times one parameters set
            while(n>0 and time.time()-begin < self.timelimit): #While there are still parameters and timelimit is not exceeded
                i = random.choice(inds[mask]) # One new index is randomly chosen
                mask[i]=False #It wont be chosen again
                param = self.projfunctions[p][i] #Select the corresponding parameters set
                n=n-1 #Decrease the counter
                X_p = self.projections.proj(self.X, p, param) #Apply the projection
                y_pred, err = testhyp(self.hs, X_p, self.y, D) #Apply the source hypothesis
                if err < err_min: #If bette than previously, this error rate becomes the best rate and keep the related parameters in memory
                    err_min = err
                    param_min = param
                    y_pred_min = y_pred
                    p_min = p
                if err_min < self.threshold: #If we did better than self.threshold, we stop searching parameters
                    break
            if err_min < self.threshold: #We also stop searching other functions
                break
        self.projections.add(p_min,param_min) #We add the selected projection to the selected projection set
        self.lastscore=err_min #Update the last score
        return y_pred_min, err_min
    
    def getProjections(self): #Returns an object encapsulating the selected projection set
        return self.projections
    
    def printProjections(self): #Print projections
        self.projections.printProjections()
        
class Projections:
    """The projections set class for stupid and random modes.
        Not used in neural mode (which is not allowed in python2.7)
        """
    def __init__(self,source_hypothesis=None):
        """
        Constructor
        
        Parameters
        ---------
        source_hypothesis : [n_examples][l_features] -> [n_examples]
        """
        self.projections=[] #Functions list
        self.params=[] #Parameters list
        self.hs=source_hypothesis

    def add(self,p, par): #Adds a new parameter (~function+parameters) to the list
        self.projections.append(p)
        self.params.append(par)
    
    def proj(self,X,p,param):
        """
        Apply a given projection to given target data
        
        Parameters
        ----------
        X : [n_examples][l1_features]
        p : [l1_features], [n_params] -> [l2_features]
        param : [n_params]
        
        Returns
        --------
        Xp : [n_examples][l2_features]
        """
        lis=[]
        for i in range(X.shape[0]): #For each example
            lis.append(p(X[i,:],param)) #Apply the projection
        Xp=np.array(lis)
        return Xp
    
    def labelsList(self,X):
        """
        Apply the projection list on given data, then apply the source hypothesis
        
        Parameters
        ----------
        X : [n_examples][l1_features]
        
        Returns
        --------
        yl : [n_projs][n_examples]
        """
        yl=[]
        for i in range(len(self.projections)):
            X_p = self.proj(X, self.projections[i], self.params[i])
            yl.append(self.hs(X_p))
        return yl
    
    def keepLast(self): #Keeps only the last projection found. Useful if this projection has a null error
        self.projections=self.projections[-1:]
        self.params=self.params[-1:]
        
    def printProjections(self): #Prints the projection list
        print("Projections :")
        for i in range(len(self.projections)):
            print(str(self.projections[i].__name__), str(self.params[i]))