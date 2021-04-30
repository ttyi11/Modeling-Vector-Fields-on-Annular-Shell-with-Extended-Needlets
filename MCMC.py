#MCMC

"""
Posterior estimation for the parameters using Markov Chain Gibbs Sampling.
Main function: MCMC(Z, T, A, n_list)
e.g. MCMC(Z, 10000, A, [2, 3]) where A is the design matrix with needlets level 2 and 3.

"""

## load packages

import numpy as np
import scipy.linalg as sla
import scipy.special as special
from scipy.stats import invgamma, invwishart
from numpy import sqrt, sin, cos, pi
from scipy.linalg import block_diag
import time
from scipy.stats import cauchy

from scipy.stats import invgamma


### sample V with rejection sampling from Jahndrow et al (2018)

def sample_g(c_value, V):
    n = len(V)
    V = np.array(V)
    #print(n)
    index = V**2 < 0
    para = c_value/2
    sho = 0
    while(sho<20 and np.sum(index)<n):
        V_new = invgamma.rvs(a = 1, loc=0, scale=para).reshape(-1)
        u = np.random.rand(n)
        index_up = (u< V_new/(1+V_new)).reshape(-1)
        #print(index_up)
        V[index_up] = V_new[index_up]
        index = (index+index_up)>0
        sho = sho + 1
    left = np.where(index == False)[0]
    for those in left:
        V_new = invgamma.rvs(a = 1, loc = 0, scale = para[those], size = 100)
        u = np.random.rand(100) 
        index_up = (u<V_new/(1+V_new)).reshape(-1)
              #print("judge done")
        if np.sum(index_up) != 0:                  
            V[those] = V_new[index_up][0]
            index[those] = True
    if n!=sum(index):
        print(n, sum(index))
    return V

        
def sample_h1(n, nv1):
    u = np.random.rand(n)
    return np.exp(u*nv1)-1
        
    
def sample_trunexp(n, par, low, up):
    u = np.random.rand(n)
    return low - np.log(1-u+u*np.exp(-par*(up-low)))/par
    
    
    
def sample_h(c, a, b, A, I, B, lambda_2, lambda_3, n):
    nv1 = np.log(1+a/c)
    nv2 = nv1 + np.exp(-A)*(1-np.exp(A-I))/lambda_2
    nv3 = nv2 + np.exp(-I)*(1-np.exp(I-B))/lambda_3
    nv4 = nv3 + np.exp(-B)/c
    #nv = nv1 + nv2 + nv3 + nv4
    u = np.random.rand(n)*nv4
    v1 = sample_h1(sum(u<nv1), nv1)
    v2 = sample_trunexp(sum(u<nv2)-sum(u<nv1), lambda_2, a/c, 1/c)
    v3 = sample_trunexp(sum(u<nv3)-sum(u<nv2), lambda_3, 1/c, b/c)
    u4 = np.random.rand(n-sum(u<nv3))
    v4 = b/c - np.log(1-u4)
    return np.concatenate((v1, v2, v3, v4))
    
    

def sample_h_v(c, a, b, A, I, B, lambda_2, lambda_3, n):
    nv1 = np.log(1+a/c)
    nv2 = nv1 + np.exp(-A)*(1-np.exp(A-I))/lambda_2
    nv3 = nv2 + np.exp(-I)*(1-np.exp(I-B))/lambda_3
    nv4 = nv3 + np.exp(-B)/c
    u = np.random.rand(n)*nv4
    v = np.zeros(n)
    v[u<nv1] = sample_h1(sum(u<nv1), nv1[u<nv1])
    index2 = (u<nv2)&(u>nv1)
    v[index2] = sample_trunexp(sum(index2), lambda_2[index2], (a/c)[index2], 1/c[index2])
    index3 = (u<nv3)&(u>nv2)
    v[index3] = sample_trunexp(sum(index3), lambda_3[index3], 1/c[index3], (b/c)[index3])
    u4 = np.random.rand(n-sum(u<nv3))
    #print(u4, sum(u>=nv3))
    v[u>=nv3] = (b/c)[u>=nv3] - np.log(1-u4)
    return v


def sample_l(c, a, b, n):
    A, I, B = c + np.log(1+a/c), 1 + np.log(1+1/c), b + np.log(b+b/c)
    lambda_2, lambda_3 = (I - A)*c/(1-a), (B-I)*c/(b-1)
    z = sample_h(c, a, b, A, I, B, lambda_2, lambda_3, n)
    u = np.random.rand(n)
    f = c*z + np.log(1+z)
    f_L = np.zeros(n)
    index_1 = z < a/c
    index_2 = (z>= a/c)&(z < 1/c)
    index_3 = (z < b/c)&(z >= 1/c)
    index_4 = (z >= b/c)
    f_L[index_1] = np.log(1+z[index_1])
    f_L[index_2] = A + lambda_2*(z[index_2] - a/c)
    f_L[index_3] = I + lambda_3*(z[index_3] - b/c)
    f_L[index_4] = B + c*(z[index_4] - b/c)
    index = u < np.exp(-f+f_L)
    return z[index]


def sample_l_v(c, a, b):
    n = len(c)
    A, I, B = c + np.log(1+a/c), 1 + np.log(1+1/c), b + np.log(b+b/c)
    lambda_2, lambda_3 = (I - A)*c/(1-a), (B-I)*c/(b-1)
    z = sample_h_v(c, a, b, A, I, B, lambda_2, lambda_3, n)
    u = np.random.rand(n)
    f = c*z + np.log(1+z)
    f_L = np.zeros(n)
    index_1 = z < a/c
    index_2 = (z>= a/c)&(z < 1/c)
    index_3 = (z < b/c)&(z >= 1/c)
    index_4 = (z >= b/c)
    f_L[index_1] = np.log(1+z[index_1])
    f_L[index_2] = (A + lambda_2*(z - a/c))[index_2]
    f_L[index_3] = (I + lambda_3*(z - b/c))[index_3]
    f_L[index_4] = (B + c*(z - b/c))[index_4]
    index = u < np.exp(-f+f_L)
    return z[index], index


def sample_l_all(c_value, V):
    n = len(V)
    V = np.array(V)

    index = V**2 < 0
    para = c_value/2
    sho = 0
    
    while(sho<20 and np.sum(index)<n):
        V_new, index_new = sample_l_v(c_value/2, 0.2, 10)
        V[index_new] = V_new
        index = (index+index_new)>0
        sho = sho + 1
    
    left = np.where(index == False)[0]
    
    for those in left:
        V_new = sample_l(c_value[those]/2, 0.2, 10, 100)
        if len(V_new) != 0:                  
            V[those] = V_new[0]
            index[those] = True
    if n!=sum(index):
        print(n, sum(index))
    return V
    

def update(c_value, V):
    index_g = np.where(c_value>1)[0]
    index_l = np.where(c_value<=1)[0]
    V_g = V[index_g]
    V_l = V[index_l]
    c_value_g = c_value[index_g]
    c_value_l = c_value[index_l]
    V[index_g] = sample_g(c_value_g, V_g)
    V[index_l] = 1/sample_l_all(c_value_l, V_l)
    return V 


def MCMC(Z, T, A, n_list):
    """
    Input
    Z --> array: an vector of observations
    T --> scalar: number of iterations for the MCMC
    A --> array: design matrix
    n_list --> list: list of frequencies of needlets used in the estimation (matches A)
    Output
    c_keep --> array: subsamples of posterior estimation of coefficients in the Markov chain
    sigma_list --> list: list of estimation of conditional variance of coeffients corresponding to list of frequencies.
    tau --> array: subsamples of estimation of noise
    """

    ## initialize MC
    n_sample=len(Z)/3
    sigma_list = [np.ones(T) for _ in n_list]
    for sigma in sigma_list:
        sigma[0]=1
    tau=np.ones(T)
    tau[0]=1
    c_list = [np.zeros(n) for n in n_list]
    c=np.concatenate(c_list)
    c_keep = np.zeros((sum(n_list), T))
    V_list = [np.ones(n) for n in n_list]
    cur=A
    cur00=cur.T.dot(cur)
    s=0
    inver = np.zeros((sum(n_list),sum(n_list)))

    for s in range(1, T):
        
        if (s%10==1): print(f"iteration {s}, with current tau {np.sqrt(tau[s-1])}")

        ## Sample coefficients c (gamma in the paper) from posterior distribution

        for i in range(len(n_list)):
            index_l = sum(n_list[0:i])
            index_u = sum(n_list[0:i+1]) 
            inver[index_l: index_u,index_l: index_u] = np.diag(1/(V_list[i]*sigma_list[i][s-1]))

        Sigma_inv=cur00/tau[s-1]+inver
        L0=sla.cholesky(Sigma_inv).T

        z = np.random.randn(sum(n_list))
        z = z+sla.solve_triangular(L0,cur.T.dot(Z),lower=True,check_finite=False)/tau[s-1]

        c = sla.solve_triangular(L0.T,z,check_finite=False)
        c_keep[:,s]=c


        ## Sample V and sigma from posterior distribution
        
        for i in range(len(n_list)):
            index_l = sum(n_list[0:i])
            index_u = sum(n_list[0:i+1])
            c_sub = c[index_l: index_u]
            c_value = c_sub**2/sigma_list[i][s-1] 
            V_list[i] = update(c_value, V_list[i])
            sigma_list[i][s]=invwishart.rvs(df=n_list[i]-2, scale=np.sum(c_sub**2/V_list[i]), size=1, random_state=None)


        ## Sample variance of noise from posterior distribution  
      
        tau[s]=invgamma.rvs(a=(n_sample)*3/2,scale=(Z-cur.dot(c)).T.dot(Z-cur.dot(c))/2)

    c_subsample = c_keep[:,::20]
    tau = np.sqrt(tau)     
    return c_subsample, sigma_list, tau