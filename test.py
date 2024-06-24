import numpy as np
import pandas as pd

from game import Game
from permutation_sampling import PermutationSampling
from stratified_sampling import StratifiedSampling

dataset = pd.read_csv('games_titanic_classification_random_forest.csv')
game = Game(dataset)

budget = 100000
steps = [50000,100000]

permutation_sampling = PermutationSampling(game, budget, steps)
stratified_sampling = StratifiedSampling(game, budget, steps)

permutation_estimates = permutation_sampling.get_estimates()
stratified_estimates = stratified_sampling.get_estimates()
print(permutation_estimates)
print(stratified_estimates)
