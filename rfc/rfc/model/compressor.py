import pandas as pd
import random as rd

import gurobipy as gp

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate

epsilon = 10e-4

class Compressor:
    dataset : pd.DataFrame
    ensemble : Ensemble
    lazy : bool
    def __init__(
        self,
        ensemble: Ensemble,
        dataset: pd.DataFrame,
        lazy: bool = False
    ) -> None:
        if not isinstance(ensemble, Ensemble):
            raise TypeError('ensemble must be a TreeEnsemble')

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError('dataset must be a DataFrame')
        if not isinstance(lazy, bool):
            raise TypeError('lazy must be a boolean')
        self.ensemble = ensemble
        self.dataset = dataset
        self.lazy = lazy
        self.mdl = gp.Model(name = 'Compressor') # type: ignore
        self.u = [1.0 for t,_ in idenumerate(ensemble)]
    
    def build(self):
        self.mdl.setParam(gp.GRB.Param.Threads, 4)#type: ignore
        u = self.mdl.addVars(len(self.ensemble), vtype=gp.GRB.BINARY, name="u") #type: ignore
        for _,row in self.dataset.iterrows():
            klass=self.ensemble.klass(row) #type: ignore
            left_expr=gp.LinExpr() # type: ignore
            for c in range(self.ensemble.n_classes):
                if c!=klass:
                    for t,_ in idenumerate(self.ensemble):
                        left_expr.add(u[t],self.ensemble.weights[t]*(self.ensemble[t].F(row,klass)-self.ensemble[t].F(row,c)))#type: ignore
                    self.mdl.addConstr(left_expr >= epsilon)#type: ignore
        self.mdl.addConstr(u.sum() >= 1, "sum_constraint")
        self.mdl.setObjective(u.sum(),sense=gp.GRB.MINIMIZE)#type: ignore
    def add(self, rows : list[dict]):
        self.dataset = self.dataset._append(rows, ignore_index=True)#type: ignore
        u = self.mdl.getVars()
        for row in rows:
            klass=self.ensemble.klass(row) #type: ignore
            left_expr=gp.LinExpr() # type: ignore
            for c in range(self.ensemble.n_classes):
                if c!=klass:
                    for t,_ in idenumerate(self.ensemble):
                        left_expr.add(u[t],self.ensemble.weights[t]*(self.ensemble[t].F(row,klass)-self.ensemble[t].F(row,c)))#type: ignore
                    self.mdl.addConstr(left_expr >= epsilon)#type: ignore
        
    def solve(self):
        self.mdl.optimize()
        self.u = [x.X for x in self.mdl.getVars()]
        return self.u
    
    def check(self,dataset = None):
        if dataset is None : dataset = self.dataset
        for _,row in dataset.iterrows():#type: ignore
            if (self.ensemble.klass(row) != self.ensemble.klass(row,self.u)): #type: ignore
                return False
        return True
                

if __name__ == '__main__':
    print('test')