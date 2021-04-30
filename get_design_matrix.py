# GET DESIGN MATRIX

"""
Main function: phi(x, B, a, r, index_list, fre_list)
Get design matrix for given observation locations x with certain level of needlets 
e.g. phi(x, 2.0, 1.0, r, [index3, index4], [2, 3]) where index3 and index4 are the quadrature points on the sphere with frequencies 2 and 3.
"""

## load packages

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.linalg as sla
import scipy.special as special
from scipy.stats import invgamma, invwishart
from scipy.special import legendre
from numpy import sqrt, sin, cos, pi
from scipy.linalg import block_diag
import time
from numpy.polynomial import legendre as L
from scipy.stats import cauchy


def phi1(t):
    if abs(t)<1:
        return np.exp(-(1/(1-t**2)))
    else:
        return 0
    #return (abs(t)<1)*np.exp(-(1/(1-t**2)))
    
norm = integrate.quad(lambda x: phi1(x), -1, 1)[0]

def phi2(t):
    return (t>-1)*integrate.quad(lambda x: phi1(x), -2, t)[0]/norm
    #return (t>-1)*integrate(phi1(x), (x,-2, t))/norm
    
def phi3(t,B):
    x=1-2*B/(B-1)*(t-1/B)
    return ((t>=0)&(t<=1/B))+((t>1/B)&(t<1))*phi2(x)

def b(t,B):
    return sqrt(phi3(t/B,B)-phi3(t,B))

def s(j,x,index,B,a,r):
    ss1=0
    ss2=0
    inner_value,inner_value1,inner_value2=inner(x,index)
    r_matrix = r.reshape(-1,1)
    for l in np.arange(int(B**(j-1))+1,int(B**(j+1))):
        c = np.zeros(l+1)
        c[-1] = 1
        c = L.legder(c)       
        grad = L.legval(inner_value, c, tensor=True)
        ss1 = ss1 + b(l/B**j,B)*(2*l+1)/4/pi*legendre(l)(inner_value)*(a/r_matrix)**(l+2)*sqrt((2*l+1))*(l+1)*a#/sqrt((1.0**(-2*l-1)-1.2**(-2*l-1))
        ss2=ss2+b(l/B**j,B)*(grad)*(a/r_matrix)**(l+1)*(2*l+1)/4/pi*sqrt((2*l+1))#/(1.0**(-2*l-1)-1.2**(-2*l-1)))        
    pro=np.kron(ss1,np.array([[0],[0],[1]])) + np.kron(inner_value1*ss2,np.array([[0],[1],[0]]))+np.kron(inner_value2*ss2,np.array([[-1],[0],[0]])) 
    return pro

                                                              
def inner(x,y):
    ma1=np.zeros((x.shape[0],y.shape[0]))+x[:,0].reshape(x.shape[0],1)
    ma2=np.zeros((x.shape[0],y.shape[0]))+y[:,0].reshape(1, y.shape[0])
    ma_diff=np.zeros((x.shape[0],y.shape[0]))+x[:,1].reshape(x.shape[0],1)-y[:,1]
    return cos(ma1)*cos(ma2)+sin(ma1)*sin(ma2)*cos(ma_diff),-sin(ma1)*sin(ma2)*sin(ma_diff),cos(ma1)*sin(ma2)*cos(ma_diff)-sin(ma1)*cos(ma2)


def phi(x, B, a, r, index_list, fre_list):
    """
    Input 
    x --> array: latitude and longitude of observations
    B --> scalar: constant greater than 1 (use 2 in our simulation)
    a --> scalar: constant no less than 1 (use 1 in our simulation)
    r --> array: altitude of observations
    index_list --> list: a list of locations on the sphere with different resolutions. We use spherical t-designs on S2 (Womersley, 2015) as the quadrature points
    fre_list --> list: a list of level of resolutions corresponding to index_list
    Outout
    phi_list --> list: a list of design matrices corresponding to different levels of needlets
    A --> array: design matrix including all level of needlets features
    """
    phi_list = []
    for index, fre in zip(index_list, fre_list):
        phi = sqrt(4*pi/index.shape[0])*s(fre,x,index,B, a, r)
        phi_list.append(phi)
    A=np.concatenate(phi_list,axis=1)
    return phi_list, A                                                         