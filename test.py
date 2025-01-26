import numpy as np
import pandas as pd

from game import Game
from kernel_shap import KernelSHAP
from permutation_sampling import PermutationSampling
from stratified_sampling import StratifiedSampling
from stratified_svarm import StratifiedSVARM

dataset = pd.read_csv('games_titanic_classification_random_forest.csv')
game = Game(dataset)

budget = 500
steps = [100,200,500]

# define algorithms
permutation_sampling = PermutationSampling(game, budget, steps)
stratified_sampling = StratifiedSampling(game, budget, steps)
stratified_svarm = StratifiedSVARM(game, budget, steps)
kernel_shap = KernelSHAP(game, budget, steps)

# retrieve estimates
permutation_estimates = permutation_sampling.get_estimates()
stratified_estimates = stratified_sampling.get_estimates()
svarm_estimates = stratified_svarm.get_estimates()
kernel_estimates = kernel_shap.get_estimates()

print("Permutation:")
print(permutation_estimates)
print()

print("Stratified:")
print(stratified_estimates)
print()

print("Stratified SVARM:")
print(svarm_estimates)
print()

print("KernelSHAP:")
print(kernel_estimates)
