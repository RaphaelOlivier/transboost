# -*- coding: utf-8 -*-
"""
This code creates time series datasets used in other example files.
"""

from __future__ import print_function, absolute_import
import numpy as np
import os

import tools.data as data
import series.seriesGeneration as sg

def run():
    #Definition of the parameters used in dataset generation
    D = [[0.001],[0.002],[0.02], [0.2], [1]]
    sl = [1000, 700, 400, 200, 100]
    f=np.arange(10,100,1)
    r=np.arange(1,10)
    p=np.arange(0, 2*np.pi, np.pi/8)
    
    try:
        os.mkdir("seriesDATASETS")
    except:
        pass
    #Code used to generate the seriesDataset files (used in paper submissions)
    for i in range(5):
        for j in range(5):
            X,y=sg.genere_dataset(3000, 200, sinFreq=f,sinRange=r,sinPhase=p,maxSlope=sl[j],gaussianDeviation=D[i])
            data.exportCSV(X,y,"seriesDATASETS/seriesDataset_dev"+str(D[i][0])+"_slp"+str(sl[j])+".txt",delimiter='\t')
    
    try:
        os.mkdir("log")
    except:
        pass