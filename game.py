import numpy as np

class Game:
    def __init__(self, dataset):
        values = np.array(dataset.value)
        coalitions = np.array(dataset.coalition)
        self.nAttr = len(values)

        self.game = {"": values[0]}

        for index in range(1, self.nAttr):
            self.game[coalitions[index]] = values[index]

    def get_value(self, coalition):
        if len(coalition) == 0:
            return self.game[""]
        elif len(coalition) == 1:
            return self.game[str(coalition[0])]
        else:
            coalition.sort()
            name = ""
            for i in coalition:
                name += str(i) + "|"
            name = name[:-1]
            return self.game[name]