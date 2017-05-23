# -*- coding: utf-8 -*-

from testseries import testseries
import warnings
warnings.filterwarnings("ignore")


L_list=[200]
l_list=[20]
K_list=[10]

D = [[0.001],[0.002],[0.02], [0.2], [1]]
sl = [1000, 700, 400, 200, 100]

ds_list=[]
"""
for d in D:
    for s in sl:
        ds_list.append("DATASET/capDataset_dev"+str(d[0])+"_slp"+str(s)+".txt")
        
"""
#ext=[(0.001,1000),(0.001,200),(0.002,200),(0.02,200),(0.2,1000),(0.2,100)]
ext=[(0.2,100)]
for d,s in ext:
    ds_list.append("DATASET/capDataset_dev"+str(d)+"_slp"+str(s)+".txt")

#results="log/test14_2datasets_debug_projections.csv"
results=None
n=10
testseries(L_list=L_list,l_list=l_list,K_list=K_list,ds_list=ds_list,resultsFile=results,n=n)
