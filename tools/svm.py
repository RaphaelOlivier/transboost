# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 14:15:29 2015

@author: Sema Akkoyunlu

Des fonctions concernant l'apprentissage avec svm, utilisées pour construire des hypothèses sources
"""

import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
import numpy as np
import pdb
def learnSVM(X, y, gamma,logFile=None):
    """Apprentissage SVM avec validation croisée (10-fold)
    Le noyau utilisé est le noyau gaussien.
    
    Paramètres
    ---------
    X : array [n_exemples][p_features] 
    y : array [n_exemples]
    D : array [n_exemples] poids des exemples
    
    Retourne 
    ---------
    err : int
        Erreur de prédiction sur la base d'apprentissage
        
    prediction : [n_exemples]
    clf : classifieur
    
    """
    D = np.array(np.ones(len(y))/len(y))     
    clf = svm.SVC(kernel = 'rbf', C = 2, gamma = gamma)
    clf.fit(X, y)
#    prediction = cross_validation.cross_val_predict(clf, X, y, cv = 10)
    prediction = clf.predict(X)
    err = weightedError(y, prediction, D)
    return err, clf

def learnSVR(clf, X, y, L, gamma=0.03,logFile=None):
    Z=np.zeros((X.shape[0],L))
    for i in np.arange(X.shape[0]):
        x = X[i,:] 
        l=len(x)
        t = np.arange(L).reshape((L,1))
        clfr = svm.SVR(kernel = 'linear', gamma = gamma)
        clfr.fit(t[:l], x)
        Z[i,:] = np.append(x,clfr.predict(t[l:]))
        
#        if(i<5):
#            plt.plot(t[:l],X[i,:])
#            plt.plot(t,clfr.predict(t),'-')
#    prediction = cross_validation.cross_val_predict(clf, X, y, cv = 10)
    
#    plt.show()
    return testSVM(clf,Z,y)

def testSVM(clf, X, y, D = None,logFile=None):
    """ Retourne la prédiction du classifieur sur des données non vues
    
    Paramètres
    ---------
    X : array [n_examples][n_features]
    y : array [n_examples]
    
    Retourne
    ---------
    err : erreur de prédiction
    """
    if D is None:
        D = np.array(np.ones(len(y))/len(y))
        
    y_predit = clf.predict(X)
    err_test = weightedError(y, y_predit, D)
    return err_test, y_predit

def error(y, predit):
    N = len(predit)
    err=.0
    p = y * predit
    for i in range(len(p)):
        if p[i] == -1.:
            err+=1
    err/=N
    return err

def weightedError(y, y_predit, D):
    """ Le taux d'erreur est la somme pondérée des exemples mal 
    classés.
    
    Si la prédiction est bonne, le produit terme à terme des deux listes 
    est positif sinon il est négatif.
    
    Paramètres
    -------
    y : [n_exemples] 
    y_predit : [n_exemples] classes prédites par le classifieur
    *args : [n_exemples] poids attribués aux exemples
    
    Retourne 
    -------
    err : float 
        Erreur de classification
    """
    err = .0
    p = y * y_predit
    for i in range(len(p)):
        if p[i] == -1.: #Le produit est négatif si la prédiction est mauvaise
            err += D[i]
    return err