import numpy as np
import shap

class KernelSHAP():
    """This class implements KernelSHAP (Lundberg & Lee, 2017):
       Coalitions are sampled to fill the objective function of an optimization problem.
       The Shapley value is teh solution to the problem with the objective function containing all coalitions.
       Hence, solving the approximated problem exactly, approximates the Shapley values.
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
        self.n = game.get_attr()
        self.all_estimates = np.array(np.zeros((len(self.steps), self.n)))
        data = np.zeros((1, self.n))
        def model(X):
            result = np.zeros(X.shape[0])
            for i, sample in enumerate(X):
                result[i] = self.game.get_value(list(np.where(sample)[0]))
            return result
        self.explainer = shap.KernelExplainer(model=model, data=data, feature_names=np.arange(self.n))

    def get_estimates(self):
        """Approximates all Shapley values with the given budget and saves the estimates for each budget step.
            Returns:
                The estimated Shapley values for each budget step.
        """
        self.values = []
        for i in range(len(self.steps)):
            self.all_estimates[i] = self.explainer.shap_values(np.ones(self.n), nsamples=self.steps[i]-1, l1_reg=f"num_features({self.n})")
        return self.all_estimates
