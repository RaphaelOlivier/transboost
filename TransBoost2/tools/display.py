# -*- coding: utf-8 -*-
"""
Display functions used to write log files and print things
"""

from __future__ import print_function, absolute_import

def displayDict(title=None,dic={},logFile=None):
    """
    Prints the content of a dictionnary in a file, or in the shell by default
    
    Parameters
    ----------
    title : String | None (title of the file)
    dic : dict
    logFile : String (path of the file to write in, not erased if it exists) | None (no file, print in shell)
    """
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
            f.write(str(s))
        f.close()
                    
def displayList(title=None,lis=[],logFile=None):
    """
    Prints the content of a list in a file, or in the shell by default
    
    Parameters
    ----------
    title : String | None (title of the file)
    lis : list
    logFile : String (path of the file to write in, not erased if it exists) | None (no file, print in shell)
    """
    
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
            f.write(str(s)+"\n")
        f.close()