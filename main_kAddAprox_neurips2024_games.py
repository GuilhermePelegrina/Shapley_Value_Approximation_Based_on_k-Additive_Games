
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
import mod_global_kadd_shapley
from game import Game

'''
Load games already defined - Choose one of them
games_adult_classification_random_forest
games_diabetes_regression_random_forest
games_titanic_classification_random_forest
games_wine_classification_random_forest
'''
dataset = pd.read_csv('games_titanic_classification_random_forest.csv')
values = np.array(dataset.value)

print(dataset)

# create a game object
game = Game(dataset)
print(game.get_value([2]))

# Defining the number of Mote Carlo simulations
nSimul = 1

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
error_all_k5 = np.zeros((nSimul,2**nAttr - mod_global_kadd_shapley.nParam_kAdd(5,nAttr) - 50))

count2, count3, count4, count5 = 0, 0, 0, 0

for jj in range(nSimul):

    samples = np.zeros((2**nAttr,))
    samples[0] = 0
    samples[1] = 2**nAttr-1
    #samples[2:] = random.sample(range(1, 2**nAttr-1), 2**nAttr-2)
    samples[2:] = mod_global_kadd_shapley.sampling_generator(nAttr)
    
    values_samples = np.zeros((2**nAttr,))
    values_samples[0] = values[0]
    values_samples[1] = values[-1]
    values_samples[2:] = values[samples[2:].astype(int)]
    
    samples_coal = all_coal[samples.astype(int),:]
    
    matrix_transf_k1 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(1,nAttr)]
    matrix_transf_k2 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(2,nAttr)]
    matrix_transf_k3 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(3,nAttr)]
    matrix_transf_k4 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(4,nAttr)]
    matrix_transf_k5 = matrix_transf_all[:,0:mod_global_kadd_shapley.nParam_kAdd(5,nAttr)]
    
    samples_size = np.arange(nAttr+10,2**nAttr)
    
    for ii in range(len(samples_size)):
        
        
        
        error_all_k1, error_all_k2, error_all_k3, error_all_k4, error_all_k5, count2, count3, count4, count5 = mod_global_kadd_shapley.kadd_global_shapley(values_samples,samples_size,ii,jj,matrix_transf_k1,matrix_transf_k2,matrix_transf_k3,matrix_transf_k4,matrix_transf_k5,nAttr,samples,inter_true,error_all_k2,error_all_k3,error_all_k1,error_all_k4,error_all_k5,count2,count3,count4,count5)
            
        print(ii,jj)
    
'''     
plt.plot(np.arange(nAttr+10,2**nAttr), np.mean(error_all_k1,0), label='1-Additive')
plt.plot(np.arange(nParam_kAdd(2,nAttr)+20,2**nAttr), np.mean(error_all_k2,0), label='2-Additive')
plt.plot(np.arange(nParam_kAdd(3,nAttr)+30,2**nAttr), np.mean(error_all_k3,0), label='3-Additive')
plt.plot(np.arange(nParam_kAdd(4,nAttr)+40,2**nAttr), np.mean(error_all_k4,0), label='4-Additive')
plt.plot(np.arange(nParam_kAdd(5,nAttr)+50,2**nAttr), np.mean(error_all_k5,0), label='5-Additive')
#plt.ylim([0,0.05])
plt.legend()    
'''
