import random

import numpy as np

class StratifiedSVARM:
    """This class implements Stratified SVARM (Kolpaczki et al., 2024):
    The marginal contributions are splitted into two coalitions.
    The population of all coalitions is stratified by size.
    With each sampled coalition a strata mean estimate of every player is updated.
    The strata estimates of each player are aggregated to obtain Shapley estimates.
    """

    def __init__(self, game, budget, steps):
        """Args:
            game: The game that maps each coalition to a worth.
            budget: The number of times coalitions of the game can be evaluated.
            steps: List of points in time (monotonically increasing, between 1 and budget), the Shapley estimates at each step are stored.
        """
        self.game = game
        self.budget = budget
        self.remaining_budget = budget
        self.steps = steps
        self.step = 0
        self.n = game.get_attr()
        self.positive_strata = np.zeros((self.n, self.n), dtype=float)
        self.negative_strata = np.zeros((self.n, self.n), dtype=float)
        self.positive_counts = np.zeros((self.n, self.n), dtype=int)
        self.negative_counts = np.zeros((self.n, self.n), dtype=int)
        self.all_estimates = np.array(np.zeros((len(self.steps), self.n)))

    def get_estimates(self):
        """Approximates all Shapley values with the given budget and saves the estimates for each budget step.
            Returns:
                The estimated Shapley values for each budget step.
        """
        # probability distribution over sizes (2,...,nAttr-2) for sampling
        probs = [0 for i in range(self.n + 1)]
        for s in range(2, self.n - 1):
            probs[s] = 1 / (self.n - 3)

        # check whether the first budget step is already 0 and save the initialized 0 estimates
        if self.step < len(self.steps) and self.steps[self.step] == 0:
            self.all_estimates[self.step] = np.zeros(self.n)
            self.step += 1

        # initialize strata
        self.__exact_calculation()
        self.__positive_warmup()
        self.__negative_warmup()

        # sampling coalitions of sizes according to size distribution until budget runs out
        while self.remaining_budget > 0:
            s = np.random.choice(range(0, self.n + 1), 1, p=probs)
            A = np.random.choice(list(range(self.n)), s, replace=False)
            self.__update_procedure(A)

        return self.all_estimates

    # calculates all border strata exactly using only coalitions of size 0,1,n-1, and n
    def __exact_calculation(self):
        # negative 0 strata
        if self.remaining_budget > 0:
            empty_value = self.game.get_value([])
            self.remaining_budget -= 1
            for i in range(self.n):
                self.negative_strata[i][0] = empty_value
                self.negative_counts[i][0] = 1
            self.__check_steps()

        # positive n-1 strata
        if self.remaining_budget > 0:
            full_value = self.game.get_value(list(range(self.n)))
            self.remaining_budget -= 1
            for i in range(self.n):
                self.positive_strata[i][self.n-1] = full_value
                self.positive_counts[i][self.n-1] = 1
            self.__check_steps()

        for i in range(self.n):
            if self.remaining_budget > 0:
                players = list(range(self.n))
                players.remove(i)
                v_plus = self.game.get_value([i])
                self.remaining_budget -= 1
                # positive 0 strata
                self.positive_strata[i][0] = v_plus
                self.positive_counts[i][0] = 1
                # negative 1 strata
                for j in players:
                    self.negative_strata[j][1] = (self.negative_strata[j][1] * self.negative_counts[j][1] + v_plus) / (self.negative_counts[j][1] + 1)
                    self.negative_counts[j][1] += 1
                self.__check_steps()

                if self.remaining_budget > 0:
                    v_minus = self.game.get_value(players)
                    self.remaining_budget -= 1
                    # negative n-1 strata
                    self.negative_strata[i][self.n-1] = v_minus
                    self.negative_counts[i][self.n-1] = 1
                    # positive n-2 strata
                    for j in players:
                        self.positive_strata[j][self.n-2] = (self.positive_strata[j][self.n-2] * self.positive_counts[j][self.n-2] + v_minus) / (self.positive_counts[j][self.n-2] + 1)
                        self.positive_counts[j][self.n-2] += 1
                    self.__check_steps()

    # initializes all positive strata with one sample
    def __positive_warmup(self):
        for s in range(2, self.n-1):
            pi = np.random.choice(list(range(self.n)), self.n, replace=False)

            for k in range(0, int(self.n / s)):
                if self.remaining_budget > 0:
                    A = [pi[r + k * s - 1] for r in range(1, s+1)]
                    v = self.game.get_value(A)
                    self.remaining_budget -= 1
                    for i in A:
                        self.positive_strata[i][s-1] = v
                        self.positive_counts[i][s-1] = 1
                    self.__check_steps()

            if self.n % s != 0:
                if self.remaining_budget > 0:
                    A = list([pi[r - 1] for r in range(self.n - (self.n % s) + 1, self.n+1)])
                    players = [player for player in list(range(self.n)) if player not in A]
                    B = list(np.random.choice(players, s - (self.n % s), replace=False))
                    v = self.game.get_value(list(set(A + B)))
                    self.remaining_budget -= 1
                    for i in A:
                        self.positive_strata[i][s-1] = v
                        self.positive_counts[i][s-1] = 1
                    self.__check_steps()


    # initializes all negative strata with one sample
    def __negative_warmup(self):
        for s in range(2, self.n-1):
            pi = np.random.choice(list(range(self.n)), self.n, replace=False)

            for k in range(0, int(self.n / s)):
                if self.remaining_budget > 0:
                    A = [pi[r + k * s - 1] for r in range(1, s+1)]
                    players = [player for player in list(range(self.n)) if player not in A]
                    v = self.game.get_value(players)
                    self.remaining_budget -= 1
                    for i in A:
                        self.negative_strata[i][self.n-s] = v
                        self.negative_counts[i][self.n-s] = 1
                    self.__check_steps()

            if self.n % s != 0:
                if self.remaining_budget > 0:
                    A = [pi[r - 1] for r in range(self.n - (self.n % s) + 1, self.n+1)]
                    players = [player for player in list(range(self.n)) if player not in A]
                    B = list(np.random.choice(players, s - (self.n % s), replace=False))
                    players = [player for player in list(range(self.n)) if player not in list(set(A + B))]
                    v = self.game.get_value(players)
                    self.remaining_budget -= 1
                    for i in A:
                        self.negative_strata[i][self.n-s] = v
                        self.negative_counts[i][self.n-s] = 1
                    self.__check_steps()

    # updates the strata estimates with the given coalition
    def __update_procedure(self, A):
        v = self.game.get_value(A)
        self.remaining_budget -= 1
        s = len(A)
        for i in A:
            self.positive_strata[i][s-1] = (self.positive_strata[i][s-1] * self.positive_counts[i][s-1] + v) / (self.positive_counts[i][s - 1] + 1)
            self.positive_counts[i][s-1] += 1
        not_A = [i for i in list(range(self.n)) if i not in A]
        for i in not_A:
            self.negative_strata[i][s] = (self.negative_strata[i][s] * self.negative_counts[i][s] + v) / (self.negative_counts[i][s] + 1)
            self.negative_counts[i][s] += 1
        self.__check_steps()

    # aggregates the strata estimates to Shapley estimates accounting for strata with zero samples
    def __aggregate_strata(self):
        positive_halves = 1 / (np.sum(np.where(self.positive_counts > 0, 1, 0), axis=1)) * np.sum(self.positive_strata, axis=1)
        negative_halves = 1 / (np.sum(np.where(self.negative_counts > 0, 1, 0), axis=1)) * np.sum(self.negative_strata, axis=1)
        return positive_halves - negative_halves

    # check for budget step and eventually store current estimates
    def __check_steps(self):
        if self.step < len(self.steps) and self.steps[self.step] == self.budget - self.remaining_budget:
            self.all_estimates[self.step] = self.__aggregate_strata()
            self.step += 1
