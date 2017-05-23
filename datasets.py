# -*- coding: utf-8 -*-
import random
import numpy as np

import matplotlib.pyplot as plt
from data import *
import csv

"""
sinFreq=np.arrange(5, 15, 0.25)
sinRange=np.arange(1,10)
sinPhase=np.arange(0, 2*np.pi, np.pi/4)
maxSlope=1./300
gaussianDeviation=np.linspace(0.01, 0.1, 0.01)

X,y = genere_dataset(1000, 150, sinFreq=sinFreq,sinRange=sinRange,sinPhase=sinPhase,maxSlope=maxSlope,gaussianDeviation=gaussianDeviation)
exportCSV(X,y,"DATASET/generatedDataset4.txt",delimiter='\t')

"""

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
for i in range(5):
    for j in range(5):
        X,y=genere_dataset(3000, 200, sinFreq=f2,sinRange=r2,sinPhase=p,maxSlope=sl[j],gaussianDeviation=D[i])
        exportCSV(X,y,"DATASET/capDataset_dev"+str(D[i][0])+"_slp"+str(sl[j])+".txt",delimiter='\t')
"""
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                ind=str(i)+str(j)+str(k)+str(l)
                X,y=genere_dataset(900, 150, sinFreq=F[i],sinRange=R[j],sinPhase=p,maxSlope=M[k],gaussianDeviation=D[l])
                exportCSV(X,y,"DATASET/generatedDataset_"+ind+".txt",delimiter='\t')
"""

