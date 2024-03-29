import pandas as pd
import random as rd

import gurobipy as gp

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate

comp = lambda x,y : x<y
epsilon = 1.

class RelaxCompressor:
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
        self.mdl.setParam(gp.GRB.Param.Threads, 8)#type: ignore
        u = self.mdl.addVars(len(self.ensemble), vtype=gp.GRB.CONTINUOUS, name="u") #type: ignore
        for _,row in self.dataset.iterrows():
            klass=self.ensemble.klass(row,tiebreaker = comp)  #type: ignore
            left_expr=gp.LinExpr() # type: ignore
            for c in range(self.ensemble.n_classes):
                if c!=klass:
                    for t,_ in idenumerate(self.ensemble):
                        left_expr.add(u[t],self.ensemble.weights[t]*(self.ensemble[t].F(row,klass)-self.ensemble[t].F(row,c)))#type: ignore
                    if comp(c,klass):
                        self.mdl.addConstr(left_expr >= epsilon)#type: ignore
                    else:
                        self.mdl.addConstr(left_expr >= 0)#type: ignore
        self.mdl.addConstr(u.sum() >= 1, "sum_constraint")
        self.mdl.setObjective(u.sum(),sense=gp.GRB.MINIMIZE)#type: ignore
    def add(self, rows : list[dict]):
        self.dataset = self.dataset._append(rows, ignore_index=True)#type: ignore
        u = self.mdl.getVars()
        for row in rows:
            klass=self.ensemble.klass(row,tiebreaker = comp)  #type: ignore
            left_expr=gp.LinExpr() # type: ignore
            for c in range(self.ensemble.n_classes):
                if c!=klass:
                    for t,_ in idenumerate(self.ensemble):
                        left_expr.add(u[t],self.ensemble.weights[t]*(self.ensemble[t].F(row,klass)-self.ensemble[t].F(row,c)))#type: ignore
                    if comp(c,klass):
                        self.mdl.addConstr(left_expr >= epsilon)#type: ignore
                    else:
                        self.mdl.addConstr(left_expr >= 0)#type: ignore
        
    def solve(self):
        self.mdl.optimize()
        print([x.X for x in self.mdl.getVars()])
        self.u = [round(x.X) for x in self.mdl.getVars()]
        return self.u
    
    def check(self,dataset = None, rate = False):
        if dataset is None : dataset = self.dataset
        if not rate:
            for _,row in dataset.iterrows():#type: ignore
                if (self.ensemble.klass(row,tiebreaker = comp)  != self.ensemble.klass(row,self.u, tiebreaker = comp) ): #type: ignore
                    return False
            return True
        else:
            s=0
            for _,row in dataset.iterrows():#type: ignore
                s+=((self.ensemble.klass(row,tiebreaker = comp)  == self.ensemble.klass(row,self.u, tiebreaker = comp) )+0.) #type: ignore
            return s/len(dataset)

    
    def update_dataset(self, df):
        self.dataset = df
