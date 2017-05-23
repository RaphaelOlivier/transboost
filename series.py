# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created in mars 2017 

@author: Raphaël Olivier

#material for using TransBoost with time series : hypothesis and projection functions
"""

import numpy as np
import svm
from scipy.optimize import leastsq
import data

def polyline(x,param):
    L = param[0]
    d = param[1]
    f = param[2]
    b=param[3]
    a=param[4]
    l = len(x)
    t = np.linspace(1, L, L)
    A = np.vstack([np.linspace(d,f,f-d),np.ones(f-d)]).T
    m,c = np.linalg.lstsq(A, x[d:f])[0]
    x0 = m*b+c
    reg1 = lambda z : m*z+c
    reg2 = lambda z : -a*m*(z-b) + x0
    x_cont = None

    if(l<=b and b<=L):
        x_cont = np.append(np.append(x,np.apply_along_axis(reg1,0,t[l:b])),np.apply_along_axis(reg2,0,t[b:L]))
    if(b<l and l<=L):
        x_cont = np.append(x,np.apply_along_axis(reg2,0,t[l:L]))
    if(L<b and b<=l):
        x_cont = x[:L]
    if(b<=L and L<l):
        x_cont = np.append(x[:b],np.apply_along_axis(reg2,0,t[b:L]))
    return x_cont


def polyline_param(L,l):
    lis=[]
    coeff = np.arange(0,10,0.5)
    breakpoint=np.arange(L-10,10,-10)
    deb = np.arange(0,l-10,10)
    for d in deb:
        fin=np.arange(d+5,l,10)
        for f in fin:
            for b in breakpoint:
                for c in coeff:
                    lis.append((L, d, f, b, c))
    return lis


def line(x,param):
    L = param[0]
    d = param[1]
    f = param[2]
    alpha=param[3]
    l = len(x)
    t = np.linspace(1, L, L)
    A = np.vstack([np.linspace(d,f,f-d),np.ones(f-d)]).T
    m,c = np.linalg.lstsq(A, x[d:f])[0]
    m0=np.tan(np.arctan(m)+alpha)
    reg2 = lambda z : m0*z+c
    x_cont = np.apply_along_axis(reg2,0,t)

    return x_cont


def line_param(L,l):
    lis=[]
    alpha = np.arange(-np.pi/2, np.pi/2, np.pi/16)
    pace=l/5
    deb = np.arange(0,l-pace,pace)
    for d in deb:
        fin=np.arange(d+pace,l,pace)
        for f in fin:
            for a in alpha:
                lis.append((L, d, f, a))
    return lis
           
            
def breakline(x,b,a):
    L=len(x)
    t = np.linspace(1, L, L)
    A = np.vstack([t[:L],np.ones(L)]).T
    m,c = np.linalg.lstsq(A, x[:L])[0]
    x0 = m*b+c
    reg1 = lambda z : m*z+c
    reg2 = lambda z : -a*m*(z-b) + x0
    xt= x[b:] - np.apply_along_axis(reg1,0,t[b:]) +  np.apply_along_axis(reg2,0,t[b:])
    return np.append(x[:b],xt)
    
def reversesvmhyp(X,y,n):
    Xc=data.cutSeries(X,n)
    return breaksvmhyp(Xc,y,0,1)
    
def breaksvmhyp(X,y,b,c):
    X_mod=np.zeros(X.shape)
    N = X.shape[0]
    for i in range(N):
        X_mod[i,:]=breakline(X[i,:],b,c)
    
    err,clf = svm.learnSVM(X_mod, y, 0.03)
    def hs(X1):
        return clf.predict(X1)
    return hs


def svmhyp(X,y):
    
    err,clf = svm.learnSVM(X, y, 0.03)
    def hs(X1):
        return clf.predict(X1)
    return hs


def g_param_RegLin(L,l):
    deb = np.arange(0,l-10,10)
    for d in deb:
        fin=np.arange(d+5,l,10)
        for f in fin:
            yield [L, d, f]


def RegLin(x,param):
    l = len(x)
    L = param[0]
    t = np.linspace(1, L, L)
    f = param[2]
    d = param[1]
    A = np.vstack([t[d:f],np.ones(f-d)]).T
    m,c = np.linalg.lstsq(A, x[d:f])[0]
    reglin = lambda z : m*z + c
    x_cont = np.append(x,np.apply_along_axis(reglin,0,t[l:L]))
    
    return x_cont


def g_param_Sinus(L):
    A = 1
    phi = np.pi/2
    b = 0
    for freq in range(10, 800, 100):
        for phi in np.arange(0, 2*np.pi, np.pi/2):
            for A in np.arange(-10, 10, 4):
                param = [L, A, freq, phi, b]
                yield param


def Sinus(x,param):
    """ Continuateur sinus de la forme A*sin(2*pi*freq*t + phi)+ a*t + b

    param [A, freq, phi, a, b]
    """
    l = len(x)
    L = param[0]
    t = np.linspace(1, L, L)
    fitfunc = lambda p, t: p[0]*np.sin(2*np.pi*p[1]*t+p[2]) + p[3] # Target function
    errfunc = lambda p, t, x: fitfunc(p, t) - x
    p0 = param[1:]
    p1 = leastsq(errfunc, p0[:], args=(t[0:l], x))[0]
    x_c = fitfunc(p1, t)
    x_cont = np.append(x, x_c[l:L], axis =0)
    return x_cont

