# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:44:07 2017

@author: raphael

This code computes a very large set of experiences on time series. It then displays the 2-dimensonnal graph that sums up the experience. It was used in submissions.
It requires running series0.py previously.
It may be very long to compute. If you want a shorter, less complete test, see series3.py
"""

from __future__ import print_function, absolute_import

from series.seriesTesting import testseries, displayExperience

def run():
    L_list=[200]
    l_list=[20,50,100]
    K_list=[5,10]
    
    D = [[0.001],[0.002],[0.02], [0.2], [1]]
    sl = [1000, 700, 400, 200, 100]
    
    ds_list=[]
    
    for d in D:
        for s in sl:
            ds_list.append("seriesDATASETS/seriesDataset_dev"+str(d[0])+"_slp"+str(s)+".txt")
            
    results="log/series2results.csv"
    n=100
    testseries(L_list=L_list,l_list=l_list,K_list=K_list,ds_list=ds_list,resultsFile=results,n=n)
    
    #Display the graph
    df = pd.read_csv("log/series2results.csv")
    displayExperience(df,"transBoost test score", "reference test score")
