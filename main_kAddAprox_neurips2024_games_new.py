
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import mod_global_kadd_shapley
from game import Game


'''
Load games already defined - Choose one of them
games_adult_classification_random_forest
games_adult_local
games_diabetes_regression_random_forest
games_lsac_classification_random_forest
games_titanic_classification_random_forest
games_wine_classification_random_forest
games_Fifa_unsupervised
games_BreastCancer_unsupervised
games_BigFive_unsupervised
games_nlp_sentiment
games_image_cat
'''
dataset = pd.read_csv('games_image_cat.csv')
values = np.array(dataset.value)

cardinal = []
cardinal.append(0)
for ii in range(1,len(values)-1):
    cardinal.append(dataset.coalition[ii].count('|')+1)
cardinal.append(int(math.log(len(values),2)))

# Defining the number of Mote Carlo simulations
nSimul = 100

# Extracting the number of features
nAttr = int(math.log(len(values),2))

# Creating the trasnformation matrix from shapley indices to game
matrix_transf_all = mod_global_kadd_shapley.tr_shap2game(nAttr, nAttr)

# Creating a matrix whose rows contain the colaition indices for all coalitions
all_coal = mod_global_kadd_shapley.coalition_shap_kadd(nAttr,nAttr)

# Calculating the true shapley and interaction indices
inter_true = np.linalg.inv(matrix_transf_all) @ values

# Creating the error matrices for a set of k-additive models
error_all_k1 = np.zeros((nSimul,2**nAttr - nAttr - 10))
error_all_k2 = np.zeros((nSimul,2**nAttr - mod_global_kadd_shapley.nParam_kAdd(2,nAttr) - 20))
error_all_k3 = np.zeros((nSimul,2**nAttr - mod_global_kadd_shapley.nParam_kAdd(3,nAttr) - 30))
error_all_k4 = np.zeros((nSimul,2**nAttr - mod_global_kadd_shapley.nParam_kAdd(4,nAttr) - 40))


samples = np.zeros((nSimul,2**nAttr))

shapley_all_k1 = []
shapley_all_k2 = []
shapley_all_k3 = []
shapley_all_k4 = []

for jj in range(nSimul):

    
    samples[jj,0] = 0
    samples[jj,1] = 2**nAttr-1
    #samples[jj,2:] = random.sample(range(1, 2**nAttr-1), 2**nAttr-2)
    samples[jj,2:] = mod_global_kadd_shapley.sampling_generator(nAttr)
    
    values_samples = np.zeros((2**nAttr,))
    values_samples[0] = values[0]
    values_samples[1] = values[-1]
    values_samples[2:] = values[samples[jj,2:].astype(int)]
    
    samples_coal = all_coal[samples[jj,:].astype(int),:]
    
    matrix_transf_k1 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(1,nAttr)]
    matrix_transf_k2 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(2,nAttr)]
    matrix_transf_k3 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(3,nAttr)]
    matrix_transf_k4 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(4,nAttr)]
    
    # Selecting the number of samples (all of them or according to a budget)
    
    #samples_size = np.arange(nAttr+10,2**nAttr)
    samples_size = np.arange(nAttr+10,(1/2)*(2**nAttr)) # Define the budget
    samples_size = np.concatenate((np.arange(nAttr+10,100,1),np.arange(100,500,10),np.arange(500,2**nAttr,100)),axis=0) # Define the budget
    samples_size = np.append(samples_size,2**nAttr-1)
    
    index_ii = 0
    
    index_k1 = []
    index_k2 = []
    index_k3 = []
    index_k4 = []
    count2, count3, count4, count5 = 0, 0, 0, 0
    
    print(jj)
    
    for ii in range(len(samples_size)):
        
        error_all_k1, error_all_k2, error_all_k3, error_all_k4, count2, count3, count4, shapley_k1, shapley_k2, shapley_k3, shapley_k4 = mod_global_kadd_shapley.kadd_global_shapley(values_samples,samples_size,ii,jj,matrix_transf_k1,matrix_transf_k2,matrix_transf_k3,matrix_transf_k4,nAttr,samples[jj,:],inter_true,error_all_k2,error_all_k3,error_all_k1,error_all_k4,count2,count3,count4,index_k1,index_k2,index_k3,index_k4,cardinal)
        
        shapley_all_k1.append(shapley_k1)
        
        if len(shapley_k2) > 0:
            shapley_all_k2.append(shapley_k2)
        if len(shapley_k3) > 0:
            shapley_all_k3.append(shapley_k3)
        if len(shapley_k4) > 0:
            shapley_all_k4.append(shapley_k4)
        
        print(ii,jj)
    
''' 
error_all_k1 = np.delete(error_all_k1, np.argwhere(np.all(error_all_k1[..., :] == 0, axis=0)), axis=1)
error_all_k2 = np.delete(error_all_k2, np.argwhere(np.all(error_all_k2[..., :] == 0, axis=0)), axis=1)
error_all_k3 = np.delete(error_all_k3, np.argwhere(np.all(error_all_k3[..., :] == 0, axis=0)), axis=1)
error_all_k4 = np.delete(error_all_k4, np.argwhere(np.all(error_all_k4[..., :] == 0, axis=0)), axis=1)
    
plt.plot(index_k1, np.mean(error_all_k1,0), label='$k = 1$')
plt.plot(index_k2, np.mean(error_all_k2,0), label='$k = 2$')
plt.plot(index_k3, np.mean(error_all_k3,0), label='$k = 3$')
plt.plot(index_k4, np.mean(error_all_k4,0), label='$k = 4$')
#plt.ylim([0,0.01])
plt.xlabel('# of value function evaluations ($T$)', fontsize=14)
plt.ylabel('Average MSE', fontsize=14)
plt.legend(fontsize=12)  
'''

data_save = [samples,error_all_k1, error_all_k2, error_all_k3, error_all_k4, index_k1, index_k2, index_k3, index_k4, shapley_k1, shapley_k2, shapley_k3, shapley_k4]
np.save('results_kAddApprox_test.npy', data_save, allow_pickle=True)
#samples,error_all_k1, error_all_k2, error_all_k3, error_all_k4, index_k1, index_k2, index_k3, index_k4, shapley_k1, shapley_k2, shapley_k3, shapley_k4 = np.load('results_kAddApprox_wine_rf.npy', allow_pickle=True)
