import pandas as pd
import numpy as np
import random

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate

class GreedyCompressor:
    ensemble : Ensemble
    dataset : pd.DataFrame

    def __init__(self, ensemble: Ensemble, dataset: pd.DataFrame) -> None:
        self.ensemble = ensemble
        self.dataset = dataset
        self.u = np.ones(len(self.ensemble))
        self.p = np.vstack(tuple(self.dataset.apply(self.ensemble.p, axis=1).values))

    def solve(self,iterations=100, tuples=1, choices = None):
        if tuples<1:
            raise TypeError('tuples has to be a postive integer.')
        if choices is None:
            choices = int(len(self.ensemble)**1.5)
        for _ in range(iterations):
            u = np.ones(len(self.ensemble))
            p = self.p.copy()
            k = np.argmax(p, axis=1)
            idx = list(range(len(self.ensemble)))
            random.shuffle(idx)
            for i in range(tuples,1,-1):
                for choice in range(choices):
                    couple = np.random.choice(np.nonzero(self.u), i, replace = False)
                    T=[self.ensemble[t] for t in couple]
                    pp = []
                    for c in range(self.ensemble.n_classes):
                        ppp = self.dataset.apply(lambda x: sum([t.F(x, c) for t in T]), axis=1).values#type:ignore
                        pp.append(ppp)
                    pp = np.vstack(tuple(pp)).T
                    kk = np.argmax(p - pp, axis=1)
                    if (k == kk).all():
                        for t in couple:
                            u[t] = 0.
                            p -= pp
                    
            for t in idx:
                T = self.ensemble[t]
                pp = []
                for c in range(self.ensemble.n_classes):
                    ppp = self.dataset.apply(lambda x: T.F(x, c), axis=1).values #type:ignore
                    pp.append(ppp)
                pp = np.vstack(tuple(pp)).T
                kk = np.argmax(p - pp, axis=1)
                if (k == kk).all():
                    u[t] = 0.
                    p -= pp
                if sum(u) < sum(self.u):
                    self.u = u
