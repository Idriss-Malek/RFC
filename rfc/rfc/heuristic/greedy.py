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

    def solve(self):
        p = self.dataset.apply(self.ensemble.p, axis=1).values
        p = np.vstack(tuple(p))
        k = np.argmax(p, axis=1)
        idx = list(range(len(self.ensemble)))
        random.shuffle(idx)
        r = []
        for t in idx:
            T = self.ensemble[t]
            pp = []
            for c in range(self.ensemble.n_classes):
                ppp = self.dataset.apply(lambda x: T.F(x, c), axis=1).values
                pp.append(ppp)
            pp = np.vstack(tuple(pp)).T
            kk = np.argmax(p - pp, axis=1)
            if (k == kk).all():
                r.append(t)
                p -= pp
        print(len(r))