import numpy as np

class PermuationSampling:
    """This class implements ApproShapley (Castro et al., 2009):
    Permutations of the player set are sampled and the marginal contribution of neighboring players within a permutation extracted to update a mean estimate of the Shapley values.
    """

    def get_estimates(self, game, budget, steps):
        """Approximates all Shapley values with the given budget and saves the estimates for each budget step.

            Args:
                game: The game that maps each coalition to a worth.
                budget: The number of times coalitions of the game can be evaluated.
                steps: List of points in time (monotonically increasing, between 1 and budget), the Shapley estimates at each step are stored.

            Returns:
                The estimated Shapley values for each budget step.
        """
        rng = np.random.default_rng()
        nAttr = game.get_attr()
        all_estimates = np.array(np.zeros((len(steps), nAttr)))
        counts = np.zeros(nAttr)
        remaining_budget = budget
        step = 0

        # check whether the first budget step is already 0 and save the initialized 0 estimates
        if step < len(steps) and steps[step] == 0:
            all_estimates[step] = np.zeros(nAttr)
            step += 1

        sums = np.zeros(nAttr)
        # iterate and generate permutations while remaining budget left
        while remaining_budget > 0:
            permutation = rng.permutation(nAttr)

            # evaluate the empty set
            old_coalition = []
            old_value = game.get_value(old_coalition)
            remaining_budget -= 1

            # check for budget step
            if step < len(steps) and steps[step] == budget - remaining_budget:
                all_estimates[step] = np.divide(sums, counts, where=counts != 0)
                step += 1

            index = 1
            while remaining_budget > 0 and index < nAttr:
                player = permutation[index]

                # evaluate the next coalition and compute update with the marginal
                new_coalition = old_coalition.copy()
                new_coalition.append(player)
                new_value = game.get_value(new_coalition)
                marginal = new_value - old_value
                sums[player] += marginal
                counts[player] += 1

                old_coalition = new_coalition
                old_value = new_value
                index += 1
                remaining_budget -= 1

                # check for budget step
                if step < len(steps) and steps[step] == budget - remaining_budget:
                    all_estimates[step] = np.divide(sums, counts, where=counts != 0)
                    step += 1

        return all_estimates





