import pandas as pd
import numpy as np

import gurobipy as gp

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate


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
    
    def build(self):
        u = self.mdl.addVars(len(self.ensemble), vtype=gp.GRB.BINARY, name="u") #type: ignore
        for index,row in self.dataset.iterrows():
            klass=self.ensemble.klass(row) #type: ignore
            left_expr=gp.LinExpr() # type: ignore
            for t,_ in idenumerate(self.ensemble):
                left_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,klass))#type: ignore
            for c in range(self.ensemble.n_classes):
                if c!=klass:
                    right_expr=gp.LinExpr() # type: ignore
                    for t,_ in idenumerate(self.ensemble):
                        right_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,c))#type: ignore
                self.mdl.addConstr(left_expr >= right_expr, f"c_{index}_{c}")#type: ignore
        self.mdl.addConstr(u.sum() >= 1, "sum_constraint")
        self.mdl.setObjective(u.sum(),sense=gp.GRB.MINIMIZE)#type: ignore
    def add(self, rows : list[dict]):
        klass=self.ensemble.klass(rows) #type: ignore
        self.dataset = self.dataset.append(rows, ignore_index=True)#type: ignore
        left_expr=gp.LinExpr() # type: ignore
        for row in rows:
            for t,_ in idenumerate(self.ensemble):
                left_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,klass))#type: ignore
            for c in range(self.ensemble.n_classes):
                if c!=klass:
                    right_expr=gp.LinExpr() # type: ignore
                    for t,_ in idenumerate(self.ensemble):
                        right_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,c))#type: ignore
                self.mdl.addConstr(left_expr >= right_expr, f"c_{index}_{c}")#type: ignore
        
    def solve(self):
        self.mdl.optimize()
        return [x.X for x in self.mdl.getVars()]
    
    def check(self):
        total=len(self.dataset)
        sucess=0
        for index,row in self.dataset.iterrows():
            sucess+=(self.ensemble.klass(row) == self.ensemble.klass(row,[x.X for x in self.mdl.getVars()])) #type: ignore
        return sucess/total 
                

if __name__ == '__main__':
    print('test')