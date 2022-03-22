import yaml
import numpy as np

class helpers:
    def __init__(self):
        pass

    def read_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def sig(self, x):
        return 1 / (1 + np.exp(-x))


    def reluder(self, x):
        # relu derivative
        if x.all()>0:
            return 1
        else:
            return 0

    ##Keeping track of the index to be able to get the right action
    def epsilongreedy(self, Qval_allowed, idx_allowed, epsilon):

        N_a=np.shape(Qval_allowed)[0]
        rand_value=np.random.uniform(0,1)
        rand_a=rand_value<epsilon

        if rand_a==True:
            a=np.random.choice(idx_allowed)

        else:
            best=np.argmax(Qval_allowed)
            a=idx_allowed[best]

        return a
