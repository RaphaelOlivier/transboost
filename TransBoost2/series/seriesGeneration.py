# -*- coding: utf-8 -*-
"""
Created in april 2017 

@author: Raphaël Olivier

Code and functions used to generate some time series datasets
"""

from __future__ import print_function, absolute_import
import random
import numpy as np

def genere_dataset(N, L, sinFreq=np.arange(5, 15, 0.25), sinRange=np.arange(1,10), sinPhase=np.arange(0, 2*np.pi, np.pi/4), maxSlope=1./300, gaussianDeviation=np.linspace(0.01, 0.05, 5)):
    """
    Creates a dataset with N times series of L values each
    And an array with their label: -1 or 1
    
    Example:
    --------
    genere_dataset(3, 10, 1)
    -> (array([[-0.96, -0.94, -1.03, -0.92, -1.04, -0.98, -1.02, -0.99, -0.94,
         -1.04],
        [ 1.06,  1.03,  1.01,  1.04,  1.02,  1.01,  1.06,  0.92,  0.98,
          0.98],
        [ 0.71,  0.7 ,  0.7 ,  0.7 ,  0.72,  0.72,  0.7 ,  0.71,  0.72,
          0.73]]), array([ 1., -1.,  1.]))    
    """
    random.seed()
    X = np.zeros((N, L))  # creates an array of N lines with L colums of zeros
    y = np.zeros(N)       # create an array with N zeros
    t = np.linspace(1, L, L)  # an array with L values [1., 2. , ... L.]
    
    for i in range(N):    # for every time series
        # choose its parameters (controlling the shape)
        freq = random.choice(sinFreq)
        a=random.choice(sinRange)
        # freq is taken randomly inside [250, 275, 280, ..., 825]
        phi = random.choice(sinPhase)
        # initial phase
        b = ((-1)**i*np.random.ranf())/maxSlope
        sigma = np.random.choice(gaussianDeviation)

        l = -1
        if i%2 == 0: #Decision rule for class attribution
            l = 1
        # if the trend a is positive, the label is +1
            
        x = a*np.array(np.sin(2 * np.pi * freq * t + phi)
        + (b * t) + np.random.normal(0, sigma, L))/10
        # a sinusoïdal time series with a trend given by a*t and noise 
        X[i] = x
        y[i] = l
    return X, y




def cutSeries(data, l):
    """
    Cut a time series and get only the l first values
    
    Example:
    --------
    >>> db = genere_dataset(3, 10, 1)
    >>> db
    -> (array([[-1.  , -1.03, -1.03, -1.09, -1.  , -0.92, -0.94, -1.03, -1.02,
         -1.07],
        [ 0.93,  1.03,  0.99,  1.01,  1.  ,  1.  ,  1.01,  0.97,  1.01,
          0.99],
        [ 0.72,  0.69,  0.69,  0.66,  0.72,  0.68,  0.7 ,  0.66,  0.74,
          0.75]]), array([ 1., -1.,  1.]))
    >>> db_0 = db[0]
    >>> db_0
    -> array([[-1.  , -1.03, -1.03, -1.09, -1.  , -0.92, -0.94, -1.03, -1.02,
        -1.07],
       [ 0.93,  1.03,  0.99,  1.01,  1.  ,  1.  ,  1.01,  0.97,  1.01,
         0.99],
       [ 0.72,  0.69,  0.69,  0.66,  0.72,  0.68,  0.7 ,  0.66,  0.74,
         0.75]]) 
    >>> tronquer_base(db_0, 3)
    -> array([[-1.  , -1.03, -1.03],
       [ 0.93,  1.03,  0.99],
       [ 0.72,  0.69,  0.69]])
    """
    if l < len(data[0,:]):
        return data[:, 0:l]
    return data


#Definition of the parameters used in dataset generation
f1=np.arange(1,10,0.1)
f2=np.arange(10,100,1)
f3=np.arange(100,1000,10)
f4=f1+f2+f3
F=[f1,f2,f3,f4]
r1=np.arange(0.1,1,0.1)
r2=np.arange(1,10)
r3=np.arange(10,100,10)
r4=r1+r2+r3
R=[r1,r2,r3,r4]
p=np.arange(0, 2*np.pi, np.pi/8)
m1= 10
m2= 50
m3= 200
m4=1000
M=[m1,m2,m3,m4]
d1=np.arange(0.001, 0.01, 0.001)
d2=np.arange(0.01, 0.1, 0.01)
d3=np.arange(0.1, 1., 0.1)
d4=d1+d2+d3
D=[d1,d2,d3,d4]
D = [[0.001],[0.002],[0.02], [0.2], [1]]
sl = [1000, 700, 400, 200, 100]
f2=np.arange(10,100,1)
r2=np.arange(1,10)
p=np.arange(0, 2*np.pi, np.pi/8)


"""
#Code used to generate the capDataset files
for i in range(5):
    for j in range(5):
        X,y=data.genere_dataset(3000, 200, sinFreq=f2,sinRange=r2,sinPhase=p,maxSlope=sl[j],gaussianDeviation=D[i])
        data.exportCSV(X,y,"DATASET/capDataset_dev"+str(D[i][0])+"_slp"+str(sl[j])+".txt",delimiter='\t')
"""

"""
#Code used to generate the genere_dataset files
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                ind=str(i)+str(j)+str(k)+str(l)
                X,y=genere_dataset(900, 150, sinFreq=F[i],sinRange=R[j],sinPhase=p,maxSlope=M[k],gaussianDeviation=D[l])
                exportCSV(X,y,"DATASET/generatedDataset_"+ind+".txt",delimiter='\t')
"""

