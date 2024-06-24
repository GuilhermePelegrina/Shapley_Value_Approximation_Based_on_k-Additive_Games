import numpy as np
import pandas as pd

from game import Game
from permutation_sampling import PermuationSampling

dataset = pd.read_csv('games_titanic_classification_random_forest.csv')
game = Game(dataset)

budget = 1000
steps = [100,200,300,400,500,600,700,800,900,1000]

permutation_sampling = PermuationSampling()
all_estimates = permutation_sampling.get_estimates(game, budget, steps)

print(all_estimates)
