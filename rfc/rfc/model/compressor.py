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
        u = self.mdl.addVars(len(self.ensemble), vtype=gp.GRB.BINARY) #type: ignore
        for index,row in self.dataset.iterrows():
            left_expr=gp.LinExpr() # type: ignore
            for t,_ in idenumerate(self.ensemble):
                left_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,klass))#type: ignore
            klass=self.ensemble.klass(row) #type: ignore
            for c in range(self.ensemble.n_classes):
                if c!=klass:
                    right_expr=gp.LinExpr() # type: ignore
                    for t,_ in idenumerate(self.ensemble):
                        right_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,c))#type: ignore
                self.mdl.addConstrs(left_expr >= right_expr, f"c_{index}_{c}")#type: ignore
        self.mdl.setObjective(sum(u),sense=gp.GRB.MINIMIZE)#type: ignore
    
    def add(self, row : dict):
        self.dataset = self.dataset.append(row, ignore_index=True)#type: ignore
        left_expr=gp.LinExpr() # type: ignore
        for t,_ in idenumerate(self.ensemble):
            left_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,klass))#type: ignore
        klass=self.ensemble.klass(row) #type: ignore
        for c in range(self.ensemble.n_classes):
            if c!=klass:
                right_expr=gp.LinExpr() # type: ignore
                for t,_ in idenumerate(self.ensemble):
                    right_expr.add(u[t],self.ensemble.weights[t]*self.ensemble[t].F(row,c))#type: ignore
            self.mdl.addConstrs(left_expr >= right_expr, f"c_{index}_{c}")#type: ignore
    
    def solve(self):
        self.mdl.optimize()

if __name__ == '__main__':
    print('test')