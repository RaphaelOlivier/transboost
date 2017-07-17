# -*- coding: utf-8 -*-
"""
Created in may 2017 based on a Sema Akkoyunlu file from 2015

@author: Raphaël Olivier
"""
from __future__ import print_function
from __future__ import division
import numpy as np
from tools.data import *
from operator import itemgetter
import collections
import time
import matplotlib.pyplot as plt
def boosting(X, y, hs, K, projFinder):
    """ Calcule à chaque étape du boosting les coefficients actualisés d'Adaboost. La recherche du projecteur est encapsulée dans ProjFinder.search()

    Paramètres
    --------
    X : [n_exemples][l1_features]
    y : [n_exemples]
    hs : [n_exemples][l2_features] -> [n_exemples]
    K : int
        Nombre d'itération pour le boosting
    projFinder : ProjFinder
        Classe de recherche des projecteurs de type [l1_features] -> [l2_features]

    Retourne
    -------
    projs : Projections() classe contenant les projecteurs [l1_features] -> [l2_features]
        les projecteurs retenus par le boosting.
    e : liste ordonnée des erreurs des projecteurs
    a : liste ordonnée des coefficients à attribuer au score de chaque projecteur
    t : temps d'exploration requis pour trouver chaque projecteur
    """
    N = X.shape[0] # nombre d'exemples
    D = np.array(np.ones(N)/N) # Poids des exemples initialisés à 1/N
    a = []
    t = []
    e = []
    projFinder.init(X,y,hs) # On initialise projFinder avec les données d'entraînement et l'hypothèse source
    for k in xrange(K): # à chaque étape
        begin=time.time()
        print('Boosting : etape',k)
        y_pred,err = projFinder.search(D) # On cherche un projecteur (paramètres de recherches dépendent de projFinder)
        end=time.time()
        if(err==None): # Si aucun projecteur satisfaisant n'est trouvé, on s'arrête là.
            print("Aucun continuateur performant : interruption du boosting après "+str(k)+" étapes")
            break
        if err==0: # Si un projecteur sans erreur est trouvé, on ne garde que lui et on interrompt le boosting
            projFinder.keepLast()
            a=[1] # un seul coefficient
            e.append(0) #erreur nulle
            t.append(end-begin) #temps d'exploration
            break
        alpha = 0.5*np.log((1-err)/err) # Cas général : on calcule le coefficient à attribuer au projecteur

        D = computeWeights(D, y, y_pred, err) # On réactualise le poids de chaque exemple
        
        a.append(alpha)
        t.append(end-begin)
        e.append(err)
        
    projs = projFinder.getProjections() # Fin du boosting : on récupère les projecteurs dans une classe Projection.
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