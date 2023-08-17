import pandas as pd
import numpy as np
import random

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate

eps = 10e-3

class GreedyCompressor:
    ensemble : Ensemble
    dataset : pd.DataFrame



    def __init__(self, ensemble: Ensemble, dataset: pd.DataFrame) -> None:
        self.ensemble = ensemble
        self.dataset = dataset
        self.u = np.ones(len(self.ensemble))
        #self.p = np.vstack(tuple(self.dataset.apply(self.ensemble.p, axis=1).values))
        self.klass = {index: self.ensemble.klass(x) for index,x in self.dataset.iterrows()}#type:ignore
        self.accs = {}
        self.p = {index: [0,0] for index,x in self.dataset.iterrows()}#type:ignore

    def accur(self,id):
        self.accs[id] = 0
        for index, x in self.dataset.iterrows():
            if self.ensemble[id].klass(x) == self.klass[index]:#type:ignore
                self.accs[id] += 1
    def misclass(self, tree_id):
        res = 0
        for index,x in self.dataset.iterrows():
            klass = self.klass[index]
            if self.p[index][klass] >= self.p[index][1-klass]+ eps * klass and self.ensemble[tree_id].klass(x) == klass:#type:ignore
                res += 1
        return res

    def check(self,id):
        klass = self.klass[id]
        if self.p[id][klass] >= self.p[id][1-klass]+ eps * klass:
            return True
    

    def solve(self):
        trees = []
        idx = list(range(len(self.ensemble)))
        self.u = np.zeros(len(self.ensemble))
        while not self.dataset.index.map(self.check).all():
            t = max(idx,key = self.misclass)
            idx.remove(t)
            for index,x in self.dataset.iterrows():
                self.p[index][self.ensemble[t].klass(x)] += 1
            print(len(idx))
            self.u[t] = 1

    def solve1(self,iterations=100, tuples=1, choices = None):
        if tuples<1:
            raise TypeError('tuples has to be a postive integer.')
        if choices is None:
            choices = int(len(self.ensemble)**1.5)
        for _ in range(iterations):
            u = np.ones(len(self.ensemble))
            p = self.p.copy()
            k = np.argmax(p, axis=1)
            idx = list(range(len(self.ensemble)))
            for t in idx:
                self.accur(t)
            idx.sort(key = lambda id:self.accs[id])
            print('done')
                    
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
            for i in range(2,tuples + 1):
                for choice in range(choices):
                    try:
                        couple = np.random.choice(np.nonzero(self.u)[0], i, replace = False)
                    except:
                        break
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
        
    
        


