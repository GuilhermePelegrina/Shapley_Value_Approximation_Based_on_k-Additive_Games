# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:41:38 2024

@author: guipe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from math import comb
from scipy.special import bernoulli

from itertools import chain, combinations
import itertools

import scipy.special
import random
import sklearn.datasets 
import xgboost
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import accuracy_score
import math



def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

    
def powerset(iterable,k_add):
    '''Return the powerset (for coalitions until k_add players) of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(k_add+1))


def tr_shap2game(nAttr, k_add):
    '''Return the transformation matrix from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(k_add) #Números de Bernoulli
    k_add_numb = nParam_kAdd(k_add,nAttr)
    total_numb = nParam_kAdd(nAttr,nAttr)
    
    coalit = np.zeros((total_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),nAttr)):
        s = list(s)
        coalit[i,s] = 1
        
    matrix_shap2game = np.zeros((total_numb,k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2,:]))
            aux3 = int(sum(coalit[i,:] * coalit[i2,:]))
            aux4 = 0
            for i3 in range(int(aux3+1)):
                aux4 += comb(aux3, i3) * nBern[aux2-i3]
            matrix_shap2game[i,i2] = aux4
    return matrix_shap2game

def coalition_shap_kadd(k_add,nAttr):
    ''' Return the matrix whose rows represent coalitions of players for cardinality at most k_add '''
    k_add_numb = nParam_kAdd(k_add,nAttr)
    coal_shap = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),k_add)):
        s = list(s)
        coal_shap[i,s] = 1
    return coal_shap

def shapley_kernel(M,s):
    ''' Return the Kernel SHAP weight '''
    if s == 0 or s == M:
        return 100000
    return (M-1)/(scipy.special.binom(M,s)*s*(M-s))

def sampling_generator(nAttr):
    
    probab = np.ones((nAttr,))*shapley_kernel(nAttr,1)
    for ii in range(2,nAttr):
        probab = np.concatenate((probab,np.ones((comb(nAttr,ii),))*shapley_kernel(nAttr,ii)),axis=0)
    
    sampling = np.zeros((len(probab),))
    probab_aux = probab/sum(probab)
    for ii in range(len(probab)):
        sampling[ii] = np.random.choice(np.arange(len(probab))+1, size=1, replace=False, p=probab_aux)
        probab[int(sampling[ii]-1)] = 0
        probab_aux = probab/sum(probab)
        
    sampling = sampling.astype(int)
    
    return sampling

def solver_shapley(matrix_transf,samples,ii,W,values_aux,inter_true):
    samples_select_matrix = matrix_transf[samples.astype(int),:]
    samples_select_matrix_aux = samples_select_matrix[0:ii,:]
    A = W @ samples_select_matrix_aux
    b = W @ values_aux
        
    # Solve LS problem
    x_sol = np.linalg.lstsq(A, b, rcond=None)[0]
    shapley = x_sol[1:nAttr+1]    
    #inter = x_sol[1:int(nAttr*(nAttr+1)/2+1)]
    
    error = np.sum((shapley - inter_true[1:nAttr+1])**2)
    
    return shapley, error

# Load games already defined "
dataset = pd.read_csv('games_titanic_classification_random_forest.csv')
values = np.array(dataset.value)

# Extracting parameters
nAttr = int(math.log(len(values),2))


matrix_transf_all = tr_shap2game(nAttr, nAttr)

all_coal = coalition_shap_kadd(nAttr,nAttr)

inter_true = np.linalg.inv(matrix_transf_all) @ values

nSimul = 1

error_all_k1 = np.zeros((nSimul,2**nAttr - nAttr - 10))
error_all_k2 = np.zeros((nSimul,2**nAttr - nParam_kAdd(2,nAttr) - 20))
error_all_k3 = np.zeros((nSimul,2**nAttr - nParam_kAdd(3,nAttr) - 30))
error_all_k4 = np.zeros((nSimul,2**nAttr - nParam_kAdd(4,nAttr) - 40))
error_all_k5 = np.zeros((nSimul,2**nAttr - nParam_kAdd(5,nAttr) - 50))

count2, count3, count4, count5 = 0, 0, 0, 0

for jj in range(nSimul):

    samples = np.zeros((2**nAttr,))
    samples[0] = 0
    samples[1] = 2**nAttr-1
    #samples[2:] = random.sample(range(1, 2**nAttr-1), 2**nAttr-2)
    samples[2:] = sampling_generator(nAttr)
    
    values_samples = np.zeros((2**nAttr,))
    values_samples[0] = values[0]
    values_samples[1] = values[-1]
    values_samples[2:] = values[samples[2:].astype(int)]
    
    samples_coal = all_coal[samples.astype(int),:]
    
    matrix_transf_k1 = matrix_transf_all[:,0:nParam_kAdd(1,nAttr)]
    matrix_transf_k2 = matrix_transf_all[:,0:nParam_kAdd(2,nAttr)]
    matrix_transf_k3 = matrix_transf_all[:,0:nParam_kAdd(3,nAttr)]
    matrix_transf_k4 = matrix_transf_all[:,0:nParam_kAdd(4,nAttr)]
    matrix_transf_k5 = matrix_transf_all[:,0:nParam_kAdd(5,nAttr)]
    
    samples_size = np.arange(nAttr+10,2**nAttr)
    
    for ii in range(len(samples_size)):
        
        values_aux = values_samples[0:samples_size[ii]]
        
        W = np.eye(len(values_aux))
        W[0,0] = 10**6
        W[1,1] = 10**6
        
        shapley_k1, error_k1 = solver_shapley(matrix_transf_k1,samples,samples_size[ii],W,values_aux,inter_true)
        error_all_k1[jj,ii] = error_k1
            
        if samples_size[ii] >= nParam_kAdd(2,nAttr)+20:
            shapley_k2, error_k2 = solver_shapley(matrix_transf_k2,samples,samples_size[ii],W,values_aux,inter_true)
            error_all_k2[jj,count2] = error_k2
            count2 += 1
            
        if samples_size[ii] >= nParam_kAdd(3,nAttr)+30:
            shapley_k3, error_k3 = solver_shapley(matrix_transf_k3,samples,samples_size[ii],W,values_aux,inter_true)
            error_all_k3[jj,count3] = error_k3
            count3 += 1
        
        if samples_size[ii] >= nParam_kAdd(4,nAttr)+40:   
            shapley_k4, error_k4 = solver_shapley(matrix_transf_k4,samples,samples_size[ii],W,values_aux,inter_true)
            error_all_k4[jj,count4] = error_k4
            count4 += 1
        
        if samples_size[ii] >= nParam_kAdd(5,nAttr)+50:
            shapley_k5, error_k5 = solver_shapley(matrix_transf_k5,samples,samples_size[ii],W,values_aux,inter_true)
            error_all_k5[jj,count5] = error_k5
            count5 += 1
            
        print(ii,jj)
    
'''     
plt.plot(np.arange(nAttr+10,2**nAttr), error_aux_k1, label='1-Additive')
plt.plot(np.arange(nParam_kAdd(2,nAttr)+20,2**nAttr), error_aux_k2, label='2-Additive')
plt.plot(np.arange(nParam_kAdd(3,nAttr)+30,2**nAttr), error_aux_k3, label='3-Additive')
plt.plot(np.arange(nParam_kAdd(4,nAttr)+40,2**nAttr), error_aux_k4, label='4-Additive')
plt.plot(np.arange(nParam_kAdd(5,nAttr)+50,2**nAttr), error_aux_k5, label='5-Additive')
#plt.ylim([0,0.05])
plt.legend()    
'''
