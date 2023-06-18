import pandas as pd
import random as rd

import gurobipy as gp

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate

epsilon = 10e-4
comp = lambda x,y : x<y

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
        self.u = {(t,v.id) : 1.0 for t,tree in idenumerate(self.ensemble) for v in tree}
        self.f = {(t,v.id,c) : 1.0 for t,tree in idenumerate(self.ensemble) for v in tree for c in self.ensemble.n_classes}
        self.u_vars = None
        self.f_vars = None
    def build(self):
        self.mdl.setParam(gp.GRB.Param.Threads, 8)#type: ignore
        self.u_vars = self.mdl.addVars([(t,v.id) for t,tree in idenumerate(self.ensemble) for v in tree], vtype=gp.GRB.BINARY, name="u") #type: ignore
        self.f_vars = self.mdl.addVars([(t,v.id,c) for t,tree in idenumerate(self.ensemble) for v in tree for c in self.ensemble.n_classes], vtype=gp.GRB.BINARY, name="f") #type: ignore
        u = self.u_vars
        f = self.f_vars
        self.mdl.addConstrs(u[(t,v.left.id)] == u[(t,v.right.id)] for t,tree in idenumerate(self.ensemble) for v in tree.nodes)
        self.mdl.addConstrs(u[(t,v.id)] <= u[(t,v.parent.id)] for t,tree in idenumerate(self.ensemble) for v in tree if tree.root.id != v.id )
        self.mdl.addConstrs(sum([f[t,v.id,c] for c in self.ensemble.n_classes]) == u[t,v.id] - u[t,v.left.id] for t,tree in idenumerate(self.ensemble) for v in tree.nodes)
        self.mdl.addConstrs(f[t,v.id,c] == u[t,v.id]*((v.klass == c)+0.) for t,tree in idenumerate(self.ensemble) for v in tree.leaves for c in self.ensemble.n_classes )
        
        self.mdl.addConstrs(sum([self.ensemble.weights[t]*f[t,v.id,self.ensemble.klass(row,tiebreaker = comp)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) >= epsilon*((comp(c,self.ensemble.klass(row,tiebreaker = comp)))+0.) + sum([self.ensemble.weights[t]*f[t,v.id,c] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) for _,row in self.dataset.iterrows() for c in self.ensemble.n_classes )
        
        self.mdl.addConstr(u.sum() >= 1, "sum_constraint")
        self.mdl.setObjective(u.sum(),sense=gp.GRB.MINIMIZE)#type: ignore
    def add(self, rows : list[dict]):
        self.dataset = self.dataset._append(rows, ignore_index=True)#type: ignore
        u = self.u_vars
        f = self.f_vars
        self.mdl.addConstrs(sum([self.ensemble.weights[t]*f[t,v.id,self.ensemble.klass(row,tiebreaker = comp)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) >= epsilon*((comp(c,self.ensemble.klass(row,tiebreaker = comp)))+0.) + sum([self.ensemble.weights[t]*f[t,v.id,c] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) for row in rows for c in self.ensemble.n_classes )
        
    def solve(self):
        self.mdl.optimize()
        self.u = [round(x.X) for x in self.u_vars]
        self.f = [round(x.X) for x in self.f_vars]
    
    def check(self,dataset = None, rate = False):
        raise NotImplementedError

    
    def update_dataset(self, df):
        self.dataset = df

if __name__ == '__main__':
    print('test')