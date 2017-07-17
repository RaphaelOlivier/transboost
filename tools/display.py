# -*- coding: utf-8 -*-
"""
Created in april 2017 
@author: Raphaël Olivier
Display functions used to write log files and print things
"""

from __future__ import print_function

def display(*args):
    logFile=args[0]
    if(logFile==None):
        for s in args[1:]:
            print(s,end='')
    else:
        for s in args[1:]:
            print(s,end='')
            logFile.write(str(s))
            
def displayDict(title=None,dic={},logFile=None):
    
    str0="----------------------------------------------------------------------------\n"
    lstr=[str0,title,"\n",str0]
    for p in dic:
        lstr=lstr+[p," : ",dic[p],"\n"]
    lstr.append("\n")
    
    if(logFile==None):
        for s in lstr:
            print(s,end='')
    else:
        f=open(logFile,'a')
        for s in lstr:
            print(s,end='')
            f.write(str(s))
        f.close()
                    
def displayList(title=None,lis=[],logFile=None):
    
    str0="----------------------------------------------------------------------------\n"
    lstr=[str0,title,"\n",str0]
    lstr=lstr+lis
    lstr.append("\n")
    
    if(logFile==None):
        for s in lstr:
            print(s,end='\n')
    else:
        f=open(logFile,'a')
        for s in lstr:
#            print(s,end='')
            f.write(str(s)+"\n")
        f.close()
        
def invDictKeys(dic):
    invdic={}
    for k in dic:
        invdic[k]=dic[k]
    return invdic