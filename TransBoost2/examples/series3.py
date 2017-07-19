# -*- coding: utf-8 -*-
"""
This code computes a set of experiences on time series. It then displays the 2-dimensonnal graph that sums up the experience.
It requires running series0.py previously.
"""

from __future__ import print_function, absolute_import
import pandas as pd

from series.seriesTesting import testseries, displayExperience
def run():
    L_list=[200]
    l_list=[20]
    K_list=[10]
    
    D = [[0.001],[0.2],[1]]
    sl = [1000, 200,100]
    
    ds_list=[]
    
    for d in D:
        for s in sl:
            ds_list.append("seriesDATASETS/seriesDataset_dev"+str(d[0])+"_slp"+str(s)+".txt")
            
    results="log/series3results.csv"
    n=5
    testseries(L_list=L_list,l_list=l_list,K_list=K_list,ds_list=ds_list,resultsFile=results,n=n)
    
    #Display the graph
    df = pd.read_csv("log/series3results.csv")
    displayExperience(df,"transBoost test score", "reference test score")
