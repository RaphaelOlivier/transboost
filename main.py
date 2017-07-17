# -*- coding: utf-8 -*-

from series.testseries import testseries
import warnings
warnings.filterwarnings("ignore")


L_list=[200]
l_list=[20]
K_list=[10]

D = [[0.001],[0.002],[0.02], [0.2], [1]]
sl = [1000, 700, 400, 200, 100]

D = [[0.001], [1]]
sl = [1000, 100]

ds_list=[]

for d in D:
    for s in sl:
        ds_list.append("DATASET/capDataset_dev"+str(d[0])+"_slp"+str(s)+".txt")
        
        
results="log/test_for_git.csv"
n=3
testseries(L_list=L_list,l_list=l_list,K_list=K_list,ds_list=ds_list,resultsFile=results,n=n)

"""
df = pd.read_csv("log/test_avec_hypothese_quelconque.csv")

testseries.displayExperience(df,"transBoost train score", "reference train score")
testseries.displayExperience(df,"transBoost test score", "reference test score")
"""