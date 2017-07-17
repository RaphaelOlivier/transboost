# -*- coding: utf-8 -*-
"""
Created in may 2017 

@author: Raphaël Olivier

Class to encapsulate transBoost and save results.
"""

from __future__ import print_function, absolute_import

from tools.display import displayDict
import transBoost.boosting as boosting

class TransBoost:
    def __init__(self):
        self.X_train, self.y_train, self.X_test, self.y_test = None,None,None,None
        self.K = 5
        self.hs=None
        self.ht=None
        self.projFinder = None
        self.log = []
        self.projections = None
        self.alphas = None
    def setProjFinder(self,p):
        self.projFinder=p
    def setSourceHyp(self,h):
        self.hs=h
    
    def setNumberSteps(self,n):
        self.K=n
    def getNumberSteps(self):
        return K
    def setTrainSet(self,X1,y1):
        self.X_train=X1
        self.y_train=y1
        
    def setTestSet(self,X1,y1):
        self.X_test=X1
        self.y_test=y1
    
        
    def getDataset(self):
        return self.X,self.y

    def checkparamslearning(self):
        if(self.X_train==None or self.y_train==None or self.hs==None or self.K==None or self.projFinder==None):
            print("param error")
            raise
    
    def checkparamstesting(self):
        if(self.X_test==None or self.y_test==None or self.hs==None or self.projections==None or self.alphas==None):
            print("param error")
            raise
    
    def learn(self):

        self.checkparamslearning()
        projections, adaboosterrors, alphas, explotime = boosting.boosting(self.X_train, self.y_train, self.hs, self.K, self.projFinder)
        self.projections=projections
        self.alphas=alphas
        log={}
        log["step"] = "Learning"
        log["data"] = {}
        log["data"]["adaboosterrors"]=adaboosterrors
        log["data"]["exploration time"]=explotime
        err, pred, trainerrors = boosting.test(self.X_train, self.y_train, self.hs, self.projections, self.alphas)
        log["results"]=err
        log["data"]["trainerrors"] = trainerrors
        self.log.append(log)
        return err
    
    def run(self):
        self.checkparamstesting()
        err, pred, testerrors = boosting.test(self.X_test, self.y_test, self.hs, self.projections, self.alphas)
        log={}
        log["step"]="Testing"
        log["data"]={}
        log["data"]["testerrors"]=testerrors
        log["results"]=err
        self.log.append(log)
        return err

    def printResults(self):
        print (self.log["step"]+" : "+str(self.log["results"]))
    
    def printLog(self,logFile=None):
        if logFile==None:
            for i in range(len(self.log)):
                print(i)
                print(self.log[i])
        else:
            for i in range(len(self.log)):
                displayDict(title="step number "+str(i),dic=self.log[i],logFile=logFile)

    def printState(self):
        dic=self.log[-1]
        dic.pop("data")
        displayDict(title="last results" ,dic=dic)
        self.projFinder.printProjections()
        
    def getLog(self):
        return self.log
    def clearLog(self):
        self.log=[]
    