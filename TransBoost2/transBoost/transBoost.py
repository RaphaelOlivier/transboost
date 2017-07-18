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
    """
    Class that encapsulates all the operations required in TransBoost application, and stores results in a log dictionnary. It allows to change at will most parameters.
    """
    def __init__(self): #Constructor
        self.X_train, self.y_train, self.X_test, self.y_test = None,None,None,None #Training and testing datasets (numpy arrays)
        self.K = 5 #Number of boosting steps
        self.projFinder = None #The object that does most of the work : the projection search. Contains informations related to the source hypothesis
        self.log = [] #The log set
        self.projections = None #The object where selected projections will be stored. Contains informations related to the source hypothesis
        self.alphas = None #The coefficients of selected projections, used to compute the final prediction
        
    def setProjFinder(self,p): #sets a projFinder (see projections.py)
        self.projFinder=p
        
    def setNumberSteps(self,n): #Sets the number of boosting steps
        self.K=n
        
    def getNumberSteps(self): #Returns the number of boosting steps
        return K
    
    def setTrainSet(self,X1,y1): #Sets the training set
        self.X_train=X1
        self.y_train=y1
        
    def setTestSet(self,X1,y1): #Sets the test set
        self.X_test=X1
        self.y_test=y1
    
    def getTrainSet(self): #Returns the training set
        return self.X_train,self.y_train
        
    def getTestSet(self): #Returns the test set
        return self.X_test,self.y_test

    def checkparamslearning(self): #Checks that all parameters are defined to start the learning phase
        if(self.X_train==None or self.y_train==None or self.K==None or self.projFinder==None):
            print("param error")
            raise
    
    def checkparamstesting(self): #Checks that all parameters are defined to start the test phase
        if(self.X_test==None or self.y_test==None or self.projections==None or self.alphas==None):
            print("param error")
            raise
            
    def checkparamsrunning(self): #Checks that all parameters are defined to run the computed projections on a dataset
        if(self.projections==None or self.alphas==None):
            print("param error")
            raise
    
    def learn(self):
        """
        Applies the TransBoost algorithm and sets the projections and coefficients. Apply them to the training set, compute final prediction and return the training error.
        """
        self.checkparamslearning() #Everything in order to execute TransBoost
        projections, adaboosterrors, alphas, explotime = boosting.boosting(self.X_train, self.y_train, self.K, self.projFinder) #Main algorithm : see boosting.py
        self.projections=projections #Selected projections
        self.alphas=alphas #Projections coefficients
        
        #logging the learning phase
        log={}
        log["step"] = "Learning"
        log["data"] = {}
        log["data"]["adaboosterrors"]=adaboosterrors
        log["data"]["exploration time"]=explotime
           
        err, pred, trainerrors = boosting.test(self.X_train, self.y_train, self.projections, self.alphas) #Apply the projections to the training set.
        
        #Logging the results
        log["results"]=err
        log["data"]["trainerrors"] = trainerrors
           
        self.log.append(log) #Save the log
        
        return err #Return the training error
    
    def test(self):
        """
        Apply the projections to the testing set, compute final prediction, return prediction and error.
        """
        self.checkparamstesting() #Everything in order to compute prediction
        err, pred, testerrors = boosting.test(self.X_test, self.y_test, self.projections, self.alphas) #Apply the projections to the test set.
        
        #loging the test phase
        log={}
        log["step"]="Testing"
        log["data"]={}
        log["data"]["testerrors"]=testerrors
        log["results"]=err
        self.log.append(log)
        return pred,err

   
    def run(self,X):
        """
        Apply the projections to the given set, compute and return final prediction.
        
        Parameters
        ----------
        X : [n_examples][l1_features]
        """
        self.checkparamsrunning() #Everything in order to compute prediction
        pred = boosting.run(self.X, self.projections, self.alphas) #Apply the projections to the given set.
        
        return pred

    def printResults(self): #Print the logged results
        print (self.log["step"]+" : "+str(self.log["results"]))
    
    def printLog(self,logFile=None):
        """
        Print everything logged, possibly in a given logFile
        
        Parameters
        ----------
        logFile : String (path to the file where log should be saved) | None (log printed in shell)
        """
        if logFile==None:
            for i in range(len(self.log)):
                print(i)
                print(self.log[i])
        else:
            for i in range(len(self.log)):
                displayDict(title="step number "+str(i),dic=self.log[i],logFile=logFile)

    def printState(self): #Print the last results
        dic=self.log[-1]
        dic.pop("data")
        displayDict(title="last results" ,dic=dic)
        self.projFinder.printProjections()
        
    def getLog(self): #Returns the log
        return self.log
    
    def clearLog(self): #Clears the log
        self.log=[]
    
