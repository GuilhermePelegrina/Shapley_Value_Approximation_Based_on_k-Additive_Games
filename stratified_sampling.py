import random

import numpy as np

class StratifiedSampling:
    """This class implements Stratified Sampling (Maleki et al., 2013):
    Each player's marginal contributions are grouped by size forming strata.
    Samples are drawn in equal frequency from all strata of all players in order to update mean estimates of the Shapley values.
    """

    def __init__(self, game, budget, steps):
        """Args:
            game: The game that maps each coalition to a worth.
            budget: The number of times coalitions of the game can be evaluated.
            steps: List of points in time (monotonically increasing, between 1 and budget), the Shapley estimates at each step are stored.
        """
        self.game = game
        self.budget = budget
        self.steps = steps

    def get_estimates(self):
        """Approximates all Shapley values with the given budget and saves the estimates for each budget step.
            Returns:
                The estimated Shapley values for each budget step.
        """
        n = self.game.get_attr()
        all_estimates = np.array(np.zeros((len(self.steps), n)))
        remaining_budget = self.budget
        step = 0

        sums = np.zeros((n, n), dtype=float)
        counts = np.zeros((n, n), dtype=int)

        # check whether the first budget step is already 0 and save the initialized 0 estimates
        if step < len(self.steps) and self.steps[step] == 0:
            all_estimates[step] = np.zeros(n)
            step += 1

        # iterate through the strata if at least two more samples are left
        while remaining_budget > 1:
            # iterate over coalition size to which a marginal contribution can be drawn
            for size in range(0, n):
                # iterate over players for whom a marginal contribution is to be drawn
                for player in range(0, n):
                    # check if enough budget is available to evaluate the first part of a marginal contribution
                    if remaining_budget > 0:
                        # generate and evaluate first coalition
                        available_players = list(range(n))
                        available_players.remove(player)
                        coalition = random.sample(available_players, size)
                        first_value = self.game.get_value(coalition)
                        remaining_budget -= 1

                        # check for budget step
                        if step < len(self.steps) and self.steps[step] == self.budget - remaining_budget:
                            all_estimates[step] = self.__aggregate_strata(sums, counts)
                            step += 1

                        # check if enough budget is available to evaluate the second part of the marginal contribution
                        if remaining_budget > 0:
                            # evaluate second coalition and compute marginal contribution
                            coalition.append(player)
                            second_value = self.game.get_value(coalition)
                            remaining_budget -= 1
                            marginal = second_value - first_value
                            sums[player][size] += marginal
                            counts[player][size] += 1

                            # check for budget step
                            if step < len(self.steps) and self.steps[step] == self.budget - remaining_budget:
                                all_estimates[step] = self.__aggregate_strata(sums, counts)
                                step += 1
        return all_estimates


    def __aggregate_strata(self, sums, counts):
        # aggregates the strata estimates to Shapley estimates accounting for strata with zero samples
        strata = np.divide(sums, counts, where=counts != 0)
        result = np.sum(strata, axis=1)
        non_zeros = np.count_nonzero(counts, axis=1)
        return np.divide(result, non_zeros, where=non_zeros != 0)


