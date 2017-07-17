# -*- coding: utf-8 -*-
"""

@author: Sema Akkoyunlu, Raphaël Olivier

Boosting core functions and a few auxiliary functions
"""
from __future__ import print_function, absolute_import, division
import numpy as np
import time

from tools.data import *

def boosting(X, y, hs, K, projFinder):
    """ Computes for each boosting step Adaboost weights. The projection reasearch part is encapsulated in ProjFinder.search()

    Parameters
    --------
    X : [n_examples][l1_features] (target examples)
    y : [n_examples] (labels)
    hs : [n_examples][l2_features] -> [n_exemples] (source hypothesis)
    K : int (number of boosting steps)
    projFinder : ProjFinder (projections research class [l1_features] -> [l2_features])

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
    projFinder.init(X,y,hs) # ProjFinder is initialised with training data and source hypothesis
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


def computeWeights(D, y, y_pred, err_min):
    """ Actualise le poids des exemples pour chaque itération du boosting

    Paramètres
    --------
    y [n_exemples]
    y_pred [n_exemples]
    D [n_exemples] poids assigné aux exemples

    Retourne
    -------
    D [n_exemples] poids actualisé

    """
    #Produit des classes attendues et prédites, si le produit est > 0 alors
    #la prédiction est bonne sinon non. On surpondère les mauvaises prédictions pour l'étape suivante
    T = y*y_pred
    rightCoeff=0.5/(1-err_min)
    wrongCoeff=0.5/err_min
    for i in range(len(T)):
        if T[i] == 1: #La prédiction est bonne
            D[i] = rightCoeff*D[i]
        else:
            D[i] = wrongCoeff*D[i]
    #La somme des coefficients n'a pas changé (1)
    return D

def weightedError(y, y_predit, D=None):
    """ Le taux d'erreur est la somme pondérée des exemples mal 
    classés.
    
    Si la prédiction est bonne, le produit terme à terme des deux listes 
    est positif sinon il est négatif.
    
    Paramètres
    -------
    y : [n_exemples] 
    y_predit : [n_exemples] classes prédites par le classifieur
    D : [n_exemples] poids attribués aux exemples
    
    Retourne 
    -------
    err : float 
        Erreur de classification
    """
    if D==None:
        N = len(y)
        D = np.array(np.ones(N)/N)
    err = .0
    p = y * y_predit
    for i in range(len(p)):
        if p[i] == -1.: #Le produit est négatif si la prédiction est mauvaise
            err += D[i]
    return err

def testhyp(h,X,y,D=None):
    """
    Applique l'hypothèse et calcule l'erreur pondérée
    
    Paramètres
    -------
    h : [n_exemples][l_features] -> [n_exemples]
    X : [n_exemples][l_features]
    y : [n_exemples] 
    D : [n_exemples] poids assigné aux exemples
    
    Retourne 
    -------
    yp : [n_exemples] prédiction
    err : float 
        Erreur de classification
    
    """
    yp=h(X)
    
    err=weightedError(y,yp,D)
    return yp,err
    
    
def test(X, y, hs, projs, alphas):
    """ Testing de projecteurs : Pour chaque projecteur on projette les données et on applique l'hypothèse h.
    On fait ensuite la somme pondérée des labels pour obtenir la prédiction finale.

    Paramètres
    -------
    X : [n_exemples][l1_features]
    y : [n_exemples] 
    hs : [n_exemples][l2_features] -> [n_exemples]
    projs : Projections, classe contenant les projecteurs [l1_features] -> [l2_features]
    alphas : [n_projecteurs]

    Retourne
    -------
    err : int
    y_pred : [n_exemples]
    """
    y_proj = projs.labelsList(X,hs) #liste des prédictions, pour chaque projecteur et l'hypothèse source h
    errprojs=[]
    y_pred = np.zeros(len(y))
    for i in range(len(alphas)):
        errprojs.append(weightedError(y, y_proj[i]))
        y_pred = y_pred+alphas[i]*y_proj[i]
    y_pred=np.sign(y_pred) #prédiction finale
    err=weightedError(y_pred,y)
    return err,y_pred, errprojs