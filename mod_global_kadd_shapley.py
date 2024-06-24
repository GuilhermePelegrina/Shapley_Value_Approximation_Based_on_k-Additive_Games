# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:14:58 2024

@author: guipe
"""


import numpy as np
from math import comb
from scipy.special import bernoulli

from itertools import chain, combinations

import scipy.special

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

def solver_shapley(matrix_transf,samples,ii,W,values_aux,inter_true,nAttr):
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

def kadd_global_shapley(values_samples,samples_size,ii,jj,matrix_transf_k1,matrix_transf_k2,matrix_transf_k3,matrix_transf_k4,nAttr,samples,inter_true,error_all_k2,error_all_k3,error_all_k1,error_all_k4,count2,count3,count4,index_k1,index_k2,index_k3,index_k4):
    
    values_aux = values_samples[0:samples_size[ii]]
    
    W = np.eye(len(values_aux))
    W[0,0] = 10**6
    W[1,1] = 10**6
    
    shapley_k1, error_k1 = solver_shapley(matrix_transf_k1,samples,samples_size[ii],W,values_aux,inter_true,nAttr)
    error_all_k1[jj,ii] = error_k1
    index_k1.append(samples_size[ii])
        
    if samples_size[ii] >= nParam_kAdd(2,nAttr)+20:
        shapley_k2, error_k2 = solver_shapley(matrix_transf_k2,samples,samples_size[ii],W,values_aux,inter_true,nAttr)
        error_all_k2[jj,count2] = error_k2
        index_k2.append(samples_size[ii])
        count2 += 1
        
    if samples_size[ii] >= nParam_kAdd(3,nAttr)+30:
        shapley_k3, error_k3 = solver_shapley(matrix_transf_k3,samples,samples_size[ii],W,values_aux,inter_true,nAttr)
        error_all_k3[jj,count3] = error_k3
        index_k3.append(samples_size[ii])
        count3 += 1
    
    if samples_size[ii] >= nParam_kAdd(4,nAttr)+40:   
        shapley_k4, error_k4 = solver_shapley(matrix_transf_k4,samples,samples_size[ii],W,values_aux,inter_true,nAttr)
        error_all_k4[jj,count4] = error_k4
        index_k4.append(samples_size[ii])
        count4 += 1
            
    return error_all_k1, error_all_k2, error_all_k3, error_all_k4, count2, count3, count4
