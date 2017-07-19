# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Material for using TransBoost with time series : hypothesis and projection functions
"""

from __future__ import print_function, absolute_import
import numpy as np
from scipy.optimize import leastsq

import tools.learning as learning
    
def svmhyp(X,y):
    """
    Create a classifier on time series (e.g. a source hypothesis)
    
    Parameters
    ----------
    X : [n_examples][L] (time series)
    y : [n_examples] (labels)
    
    Returns
    h : [m_examples][L] -> [m_examples]
    """
    
    _,clf = learning.learnSVM(X, y, 0.03) #Get classifier
    
    def h(X1): #define the hypothesis
        return clf.predict(X1)
    return h


def regLin(x,param):
    """
    linear regression projection
    
    Parameters
    ----------
    x : [l] (the series to project)
    param : [L,beg,end] (expected length of the projected series and borders of the regression)
    
    Returns
    -------
    x_proj : [L] (the projected series)
    """
    
    l = len(x)
    L = param[0]
    t = np.linspace(1, L, L) #times
    beg = param[1]
    end = param[2]
    
    A = np.vstack([t[beg:end],np.ones(end-beg)]).T #Measures on which apply linear regression
    m,c = np.linalg.lstsq(A, x[beg:end])[0] #Regression coefficients computed by least squared error
    reglin = lambda z : m*z + c #Regression
    
    x_proj = np.append(x,np.apply_along_axis(reglin,0,t[l:L])) #Projected series
    return x_proj

def regLin_param(L,l):
    """
    Generates a collection of parameters sets for linear regression projection
    
    Parameters
    ----------
    L : Integer (length of source series)
    l : Integer (length of target series)
    
    Yields
    ------
    a generator of lists [L, b, e] (b,e are the borders of the regression, varying between 0 and l)
    """
    
    beg = np.arange(0,l-10,10)
    for b in beg:
        end=np.arange(b+5,l,10)
        for e in end:
            yield [L, b, e]


def line(x,param):
    """
    linear regression projection composed with rotation
    
    Parameters
    ----------
    x : [l] (the series to project)
    param : [L,beg,end, alpha] (expected length of the projected series, borders of the regression, angle of the rotation)
    
    Returns
    -------
    x_proj : [L] (the projected series)
    """
    
    L = param[0]
    beg = param[1]
    end = param[2]
    alpha=param[3]
    t = np.linspace(1, L, L) #times
    A = np.vstack([np.linspace(beg,end,end-beg),np.ones(end-beg)]).T #Measures on which apply linear regression
    m,c = np.linalg.lstsq(A, x[beg:end])[0] #Regression coefficients computed by least squared error
    m0=np.tan(np.arctan(m)+alpha) #slope of the regression raised by angle alpha
    reg2 = lambda z : m0*z+c #Regression
    x_proj = np.apply_along_axis(reg2,0,t) #Projected series

    return x_proj

def line_param(L,l):
    """
    Generates a collection of parameters sets for "linear regression + rotation" (line) projection
    
    Parameters
    ----------
    L : Integer (length of source series)
    l : Integer (length of target series)
    
    Yields
    ------
    a collection of lists [L, b, e, a] (b,e are the borders of the regression, varying between 0 and l ; a is the angle of the rotation)
    """
    lis=[]
    alpha = np.arange(-np.pi/2, np.pi/2, np.pi/16)
    pace=l/5
    beg = np.arange(0,l-pace,pace, dtype=np.int16)
    for b in beg:
        end=np.arange(b+pace,l,pace, dtype=np.int16)
        for e in end:
            for a in alpha:
                lis.append((L, b, e, a))
    return lis


def polyline(x,param):
    """
    projection based on 2 linear functions : firstly linear regression, then linear regression composed with an enhanced, inversed slope
    
    Parameters
    ----------
    x : [l] (the series to project)
    param : [L,beg, end, j, coeff] (expected length of the projected series, borders of the regression, time of junction, -factor by which multiply the slope)
    
    Returns
    -------
    x_proj : [L] (the projected series)
    """
    L = param[0]
    beg = param[1]
    end = param[2]
    j=param[3]
    coeff=param[4]
    l = len(x)
    t = np.linspace(1, L, L) #times
    A = np.vstack([np.linspace(beg,end,end-beg),np.ones(end-beg)]).T #Measures on which apply linear regression
    m,c = np.linalg.lstsq(A, x[beg:end])[0] #Regression coefficients computed by least squared error
    x0 = m*j+c #Value at junction time
    reg1 = lambda z : m*z+c #linear regression
    reg2 = lambda z : -coeff*m*(z-j) + x0#linear regression with inversed slope enhanced by factor coeff
    x_proj = None

    #Junction with attention to different cases of unequality between L, l and j
    if(l<=j and j<=L):
        x_proj = np.append(np.append(x,np.apply_along_axis(reg1,0,t[l:j])),np.apply_along_axis(reg2,0,t[j:L]))
    if(j<l and l<=L):
        x_proj = np.append(x,np.apply_along_axis(reg2,0,t[l:L]))
    if(L<j and j<=l):
        x_proj = x[:L]
    if(j<=L and L<l):
        x_proj = np.append(x[:j],np.apply_along_axis(reg2,0,t[j:L]))
    return x_proj


def polyline_param(L,l):
    """
    Generates a collection of parameters sets for "linear regression jointed with enhanced inversed linear regression" (polyline) projection
    
    Parameters
    ----------
    L : Integer (length of source series)
    l : Integer (length of target series)
    
    Yields
    ------
    a collection of lists [L, b, e, j, c] (b,e are the borders of the regression, varying between 0 and l ; j is the junction point ; -c is the factor by which multiply the slope)
    """
    lis=[]
    coeff = np.arange(0,10,0.5)
    junctionpoint=np.arange(L-10,10,-10, dtype=np.int16)
    beg = np.arange(0,l-10,10, dtype=np.int16)
    for b in beg:
        end=np.arange(b+5,l,10, dtype=np.int16)
        for e in end:
            for j in junctionpoint:
                for c in coeff:
                    lis.append((L, b, e, j, c))
    return lis


def sinus(x,param):
    """ Sinus projection, of form A*sin(2*pi*freq*t + phi)+ a*t + b
    
    Parameters
    ----------
    x : [l] (the series to project)
    param [A, freq, phi, a, b] (initiate values of the parameters)
    
    Returns
    -------
    x_proj : [L] (the projected series)
    """
    l = len(x)
    L = param[0]
    t = np.linspace(1, L, L) #times
    fitfunc = lambda p, t: p[0]*np.sin(2*np.pi*p[1]*t+p[2]) + p[3] # projection function
    errfunc = lambda p, t, x: fitfunc(p, t) - x #Measure of the approximation error
    p0 = param[1:] #parameters of the projection function
    p1 = leastsq(errfunc, p0[:], args=(t[0:l], x))[0] #projection function after minimisation of errfunc (with updated parameters)
    x_cont = fitfunc(p1, t) #continuation of the target series
    x_proj = np.append(x, x_cont[l:L], axis =0) #projected series
    return x_proj

def sinus_param(L):
    """
    Generates a collection of parameters sets for sinus projection
    
    Parameters
    ----------
    L : Integer (length of source series)
    
    Yields
    -------
    A generator of lists [A, freq, phi, a, b], initial values of parameters in sinus projection (of form A*sin(2*pi*freq*t + phi)+ a*t + b)
    """
    A = 1
    phi = np.pi/2
    b = 0
    for freq in range(10, 800, 100):
        for phi in np.arange(0, 2*np.pi, np.pi/2):
            for A in np.arange(-10, 10, 4):
                param = [L, A, freq, phi, b]
                yield param