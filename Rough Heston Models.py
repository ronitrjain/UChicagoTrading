#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[5]:


i = complex(0, 1)


# In[8]:


#Heston Function 
def fHeston(s, St, K, r, T, sigma, kappa, theta, volvol, rho):
    #prod 
    prod = rho * sigma * i * s
    
    #Calculate d
    d1 = (prod - kappa)**2
    d2 = (sigma**2) * (i*s + s**2)
    d = np.sqrt(d1 + d2)
    
    #Calculate g 
    g1 = kappa - prod - d
    g2 = kappa - prod + d
    g = g1/g2
    
    #first exponential
    exp1 = np.exp(np.log(St) * i * s) * np.exp(i * s * r * T)
    exp2 = 1 - g * np.exp(-d * T)
    exp3 = 1 - g
    mainExpl = expl * np.power(exp2/exp3, -2*theta*kappa/(sigma **2))
    
    #second exponential
    exp4 = theta * kappa * T/(sigma **2)
    exp5 = volvol / (sigma**2)
    exp6 = (1 - np.exp(-d * T))/(1 - g * np.exp(-d * T))
    mainExp2 = np.exp((exp4 * g1) + (exp5 * g1 * exp6))
    return (mainExp1 * mainExp2)


# In[13]:


# Heston Pricer (allow for parallel processing with numba)
def priceHestonMid(St, K, r, T, sigma, kappa, theta, volvol, rho):
    P, iterations, maxNumber = 0,1000,100
    ds = maxNumber/iterations
    
    element1 = 0.5 * (St - K * np.exp(-r * T))
    
    # Calculate the complex integral
    # Using j instead of i to avoid confusion
    for j in prange(1, iterations):
        s1 = ds * (2*j + 1)/2
        s2 = s1 - i
        
        numerator1 = fHeston(s2,  St, K, r, T, 
                             sigma, kappa, theta, volvol, rho)
        numerator2 = K * fHeston(s1,  St, K, r, T, 
                              sigma, kappa, theta, volvol, rho)
        denominator = np.exp(np.log(K) * i * s1) *i *s1
        
        P = P + ds *(numerator1 - numerator2)/denominator
    
    element2 = P/np.pi
    
    return np.real((element1 + element2))


# In[11]:


# implementation of MC
def MCHeston(St, K, r, T, 
              sigma, kappa, theta, volvol, rho, 
              iterations, timeStepsPerYear):
    timeStepsPerYear = 12
    iterations = 1000000
    timesteps = T * timeStepsPerYear
    dt = 1/timeStepsPerYear
    # Define the containers to hold values of St and Vt
    S_t = np.zeros((timesteps, iterations))
    V_t = np.zeros((timesteps, iterations))
    # Assign first value of all Vt to sigma
    V_t[0,:] = sigma
    S_t[0, :] = St
    # Use Cholesky decomposition to
    means = [0,0]
    stdevs = [1/3, 1/3]
    covs = [[stdevs[0]**2          , stdevs[0]*stdevs[1]*rho], 
            [stdevs[0]*stdevs[1]*rho,           stdevs[1]**2]]
    Z = np.random.multivariate_normal(means, 
                    covs, (iterations, timesteps)).T
    Z1 = Z[0]
    Z2 = Z[1]
    for i in range(1, timesteps):
        # Use Z2 to calculate Vt
        V_t[i,:] = np.maximum(V_t[i-1,:] + 
                kappa * (theta - V_t[i-1,:])* dt + 
                volvol *  np.sqrt(V_t[i-1,:] * dt) * Z2[i,:],0)
        
        # Use all V_t calculated to find the value of S_t
        S_t[i,:] = S_t[i-1,:] + r * S_t[i,:] * dt + np.sqrt(V_t[i,:] * dt) * S_t[i-1,:] * Z1[i,:]
    return np.mean(S_t[timesteps-1, :]- K)


# In[ ]:




