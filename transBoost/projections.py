# -*- coding: utf-8 -*-
"""
Created in may 2017 

@author: Raphaël Olivier

Projection research and application
"""
from __future__ import print_function
import boosting
import numpy as np
import time
import random

class ProjFinder:
    """
    Cette classe encapsule la recherche de projecteurs, en fonction de paramètres, de données d'entraînement et d'une hypothèse source.
    Un projecteur est la donnée d'une fonction et de ses paramètres.
    """
    def __init__(self,mode="stupid", randombar=0.4999, threshold=0.45,timelimit=None):
        self.mode=mode #mode de recherche
        self.projfunctions={} #dictionnaire contenant l'ensemble des projecteurs. clés = fonctions, valeurs =collections de paramètres.
        self.randombar=randombar #limite en dessous de laquelle on estime faire mieux que le hasard (ex : 0.4999)
        self.threshold=threshold #erreur à partir de laquelle on cesse la recherche (ex : 0.45)
        self.X=None 
        self.y=None
        self.hs=None
        self.projections=Projections()
        self.lastscore=None #Dernière erreur en date (utile s'il faut cesser le boosting)
        self.timelimit=timelimit #temps maximal "caractérisque" au-delà duquel on interrompt la recherche (l'utilisation dépend du mode)
        self.graph=None
    def addFunction(self,func, params): #ajouter un couple (fonction, paramètres) à l'ensemble d'exploration
        self.projfunctions[func]=params
    
    
    
    def init(self,X,y,hs,**kwargs): #Renseigner les données d'entraînement et l'hypothèse source, initialise les projecteurs
        self.X=X
        self.y=y
        self.hs=hs
        self.projections=Projections()
        self.lastscore=None
        if(self.mode=="neural"):
           self. setGraph(**kwargs)
        
    def setGraph(**kwargs):
        if not isNN(hs):
            raise Exception("Source hypothesis should be a tensorflow graph with those parameters")
        self.graph=tf.Graph()
        
    def search(self,D):
        """
        Recherche d'un projecteur. Selon le mode, distribue la recherche à une fonction précise.
        
        Paramètres
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
                y_pred, err = boosting.testhyp(self.hs, X_p, self.y, D) #On applique l'hypothèse
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
                y_pred, err = boosting.testhyp(self.hs, X_p, self.y, D) #On applique l'hypothèse
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
    def __init__(self):
        self.projections=[] #Liste de fonctions de projecteurs
        self.params=[] #Liste de paramètres pour ces fonctions

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
    
    def labelsList(self,X, hs):
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
            yl.append(hs(X_p))
        return yl
    
    def keepLast(self): #ne conserve que le dernier projecteur trouvé. Utile si ce projecteur a une erreur nulle.
        self.projections=self.projections[-1:]
        self.params=self.params[-1:]
        
    def printProjections(self): #affiche les projecteurs
        print("Projections :")
        for i in range(len(self.projections)):
            print(str(self.projections[i].__name__), str(self.params[i]))