"""
An example of simulation: use needlets model to fit the observations

Prerequisites: 
Prepare quadrature points in ./data/
Create a folder output to save outputs of the fitting results
"""



# load package

import numpy as np
import pandas as pd
from numpy import sqrt, pi

from MCMC import MCMC
from get_design_matrix import get_design_matrix


# load locations of quadrature points

index1 = np.array(pd.read_csv('./data/index1.csv',header=None))
index2 = np.array(pd.read_csv('./data/index2.csv',header=None))
index3 = np.array(pd.read_csv('./data/index3.csv',header=None))
index4 = np.array(pd.read_csv('./data/index4.csv',header=None))
index5 = np.array(pd.read_csv('./data/index5.csv',header=None))
index6 = np.array(pd.read_csv('./data/index6.csv',header=None))


index_list = [index3, index4]
fre_list = [2.0, 3.0]
n_list = [index3.shape[0], index4.shape[0]]


# specify observatin locations (e.g.index 6 and random altitude)

index = index6.copy()

# remove singular points

range_l=index[:,0]>0.05
range_u=index[:,0]<0.95*pi
index=index[range_l&range_u,:]


n_sample = index.shape[0]
r = 1.1 * np.ones(n_sample) + 0.02 * np.random.randn(n_sample)

# get design matrix corresponding to certain level of needlets

_, A = get_design_matrix(index, 2.0, 1.0, r, index_list, fre_list)



# generate simulation coefficients

sigma_list = [1, 0.5]
spar_list = [0.8, 0.9]
k = 5   ## sigal to noise ratio

c_list = []
for n_j, sigma, spar in zip(n_list, sigma_list, spar_list):
    V = np.ones(n_j)
    V[np.random.choice(range(len(V)),int(len(V)*0.5),replace = False)] = -1
    V[np.random.choice(range(len(V)),int(len(V)*spar),replace = False)] = 0
    c = (k*sigma + sigma*np.random.randn(n_j))*V
    c_list.append(c)

c=np.concatenate(c_list)


# noise parameter

tau = 0.1

# simulate observations

Z = A.dot(c)+tau*np.random.randn(n_sample*3)



# fit the model

c_keep,sigma_list,tau = MCMC(Z, 1000, A, n_list)

Z_sample = A.dot(c_keep)

Z_est = np.mean(Z_sample[:,20:],1)

in_sample_error = np.sqrt(np.mean((Z - Z_est)**2))

print(f"Residuals from the fitted field: {in_sample_error}")


# save the fitted results

np.save("./output/c_keep", c_keep)
np.save("./output/Z_sample", Z_sample)
np.save("./output/Z_est", Z_est)


 
