# -*- coding: utf-8 -*-
"""
Created in may 2017 

@author: Raphaël Olivier

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
        threshold : float (With error below it we can stop the research instantly)
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
        Sets source hypothesis. Used in any search except neural search (does not exist in python 2.7)
        """
        self.hs=hs
    
    
    def addFunction(self,func, params):
        """
        Add a couple (function, set of parameters) in the exploration space.
        Used in stupidsearch and randomsearch.
        
        Parameters
        ---------
        func : [n_features1],[n_params] -> [n_features2]
        params [n_possible_parameters][n_sets]
        """
        self.projfunctions[func]=params
    
    def init(self,X,y):
        """
        Sets training data, initializes projections to empty space
        
        Parameters
        ----------
        X : [n_examples][n_features1]
        y : [n_examples]
        
        """
        self.X=X
        self.y=y
        self.projections=Projections(source_hypothesis=self.hs)
        self.lastscore=None
        
    def search(self,D):
        """
        Projection research. Distributes research to a precise function according to the mode
        
        Parameters
        ----------
        D : [n_exemples] poids des exemples.
        
        Retourne
        --------
        y_pred : [n_exemples] prédiction pour le projecteur retenu
        err : int, erreur pour le projecteur retenu
        """
        if(self.lastscore!=None and self.lastscore>=self.randombar): #Si le dernier trouvé n'était pas meilleur que le hasard, on indique que la recherche est terminée.
            
            return None,None
        if(self.mode=="stupid"):
            return self.stupidsearch(D)
        if(self.mode=="random"):
            return self.randomsearch(D)
        
        if(self.mode=="neural"):
            return self.neuralsearch(D)
        
    def keepLast(self): #ne conserve que le dernier projecteur trouvé. Utile si ce projecteur a une erreur nulle.
        self.projections.keepLast()
        
    def stupidsearch(self,D):
        """
        Recherche "exhaustive" d'un projecteur. On explore dans l'ordre tous les projecteurs jusqu'à ce que l'un fasse mieux que la threshold ou qu'il n'y en ait plus.
        
        Paramètres
        ----------
        D : [n_exemples] poids des exemples.
        
        Retourne
        --------
        y_pred_min : [n_exemples] prédiction pour le projecteur retenu
        err_min : int, erreur pour le projecteur retenu
        """
        err_min=2 #initialisée à une valeur max
        param_min=None
        p_min=None
        y_pred_min=None
        for p in self.projfunctions: #pour chaque fonction
            for param in self.projfunctions[p]: #pour chaque paramètre
                X_p = Projections.proj(self.X, p, param) #on applique le projecteur
                y_pred, err = testhyp(self.hs, X_p, self.y, D) #On applique l'hypothèse
                if err < err_min: #Si on a fait mieux que les scores précédents, ce score devient le score minimal et on retient les paramètres
                    err_min = err
                    param_min = param
                    y_pred_min = y_pred
                    p_min = p
#                if err_min < self.threshold: # Si on a fait mieux que threshold, on cesse la recherche
#                    break
#            if err_min < self.threshold: #On cesse ausi la recherche des fonctions
#                break
        self.projections.add(p_min,param_min) #On retient le meilleur projecteur
        self.lastscore=err_min #On retient le score
        return y_pred_min, err_min
    
    def randomsearch(self,D):
        """
        Recherche "aléatoire" d'un projecteur. On explore aléatoirement tous les projecteurs jusqu'à ce que l'un fasse mieux que la threshold, qu'o n'y en ait plus ou qu'on ait dépassé la limite de temps (pour une fonction donnée)
        La recherche aléatoire permet de s'approcher plus rapidement du score du meilleur projecteur. Avec timelimit assez élevé, la différence sera faible avec une grande probabilité.
        
        Paramètres
        ----------
        D : [n_exemples] poids des exemples.
        
        Retourne
        --------
        y_pred_min : [n_exemples] prédiction pour le projecteur retenu
        err_min : int, erreur pour le projecteur retenu
        """
        
        err_min=2 #initialisée à une valeur max
        param_min=None
        p_min=None
        y_pred_min=None
        for p in self.projfunctions: #pour chaque fonction
            begin = time.time()
            lis=self.projfunctions[p][:] #copie de la liste des paramètres
            n = len(lis)
            while(n>0 and time.time()-begin < self.timelimit): #Tant qu'on n'a pas dépassé timelimit et qu'il reste des paramètres
                i = random.randint(0,n-1) # Un indice est pris au hasard
                param = lis.pop(i) #On retire le paramètre correpondant
                n=n-1 
                X_p = Projections.proj(self.X, p, param) #On applique le projecteur
                y_pred, err = testhyp(self.hs, X_p, self.y, D) #On applique l'hypothèse
                if err < err_min: #Si on a fait mieux que les scores précédents, ce score devient le score minimal et on retient les paramètres
                    err_min = err
                    param_min = param
                    y_pred_min = y_pred
                    p_min = p
                if err_min < self.threshold: # Si on a fait mieux que threshold, on cesse la recherche
                    break
            if err_min < self.threshold: #On cesse ausi la recherche des fonctions
                break
        self.projections.add(p_min,param_min) #On retient le meilleur projecteur trouvé
        self.lastscore=err_min #On retient le score
        return y_pred_min, err_min
    
    def getProjections(self): #Retourne un objet Projections qui encapsule une liste de projecteurs
        return self.projections
    
    def printProjections(self): #affiche les projecteurs
        self.projections.printProjections()
        
class Projections:
    def __init__(self,source_hypothesis=None):
        self.projections=[] #Liste de fonctions de projecteurs
        self.params=[] #Liste de paramètres pour ces fonctions
        self.hs=source_hypothesis

    def add(self,p, par): #ajoute un couple fonction/paramètres
        self.projections.append(p)
        self.params.append(par)
    
    @staticmethod
    def proj(X,p,param):
        """
        Fonction statique : applique un projecteur sur des données.
        
        Paramètres
        ----------
        X : [n_exemples][l1_features]
        p : [l1_features], params -> [l2_features]
        param : paramètres (le type dépend de la fonction)
        
        Retourne
        --------
        Xp : [n_exemples][l2_features]
        """
        lis=[]
        for i in range(X.shape[0]): #pour chaque exemple
            lis.append(p(X[i,:],param)) #on applique le projecteur
        Xp=np.array(lis)
        return Xp
    
    def labelsList(self,X):
        """
        Applique les projecteurs sur des données et applique une hypothèse source.
        
        Paramètres
        ----------
        X : [n_exemples][l1_features]
        hs : [n_exemples][l2_features] -> [n_exemples]
        
        Retourne
        --------
        yl : [n_projecteurs][n_exemples]
        """
        yl=[]
        for i in range(len(self.projections)):
            X_p = Projections.proj(X, self.projections[i], self.params[i])
            yl.append(self.hs(X_p))
        return yl
    
    def keepLast(self): #ne conserve que le dernier projecteur trouvé. Utile si ce projecteur a une erreur nulle.
        self.projections=self.projections[-1:]
        self.params=self.params[-1:]
        
    def printProjections(self): #affiche les projecteurs
        print("Projections :")
        for i in range(len(self.projections)):
            print(str(self.projections[i].__name__), str(self.params[i]))