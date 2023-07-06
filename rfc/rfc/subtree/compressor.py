import pandas as pd
import numpy as np
import random as rd
import copy 
import gurobipy as gp

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate

epsilon = 10e-2
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
        self.f = {(t,v.id,c) : 1.0 for t,tree in idenumerate(self.ensemble) for v in tree for c in range(self.ensemble.n_classes)}
        self.u_vars = None
        self.f_vars = None
        self.compressed = None
        self.features = copy.deepcopy(self.ensemble.features)
    def build(self):
        self.mdl.setParam(gp.GRB.Param.Threads, 8)#type: ignore
        self.u_vars = self.mdl.addVars([(t,v.id) for t,tree in idenumerate(self.ensemble) for v in tree], vtype=gp.GRB.BINARY, name="u") #type: ignore
        self.f_vars = self.mdl.addVars([(t,v.id,c) for t,tree in idenumerate(self.ensemble) for v in tree for c in range(self.ensemble.n_classes)], vtype=gp.GRB.BINARY, name="f") #type: ignore
        u = self.u_vars
        f = self.f_vars
        self.mdl.addConstrs(u[(t,v.left.id)] == u[(t,v.right.id)] for t,tree in idenumerate(self.ensemble) for v in tree.nodes)
        self.mdl.addConstrs(u[(t,v.id)] <= u[(t,v.parent.id)] for t,tree in idenumerate(self.ensemble) for v in tree if tree.root.id != v.id )#type:ignore
        self.mdl.addConstrs(sum([f[(t,v.id,c)] for c in range(self.ensemble.n_classes)]) == u[t,v.id] - u[t,v.left.id] for t,tree in idenumerate(self.ensemble) for v in tree.nodes)
        self.mdl.addConstrs(f[(t,v.id,c)] == u[t,v.id]*((v.klass == c)+0.) for t,tree in idenumerate(self.ensemble) for v in tree.leaves for c in range(self.ensemble.n_classes) )
                
        for c in range(self.ensemble.n_classes):
            for _,row in self.dataset.iterrows():
                klass=self.ensemble.klass(row,tiebreaker = comp)#type:ignore
                if comp(c,klass):
                    self.mdl.addConstr(sum([self.ensemble.weights[t]*f[(t,v.id,klass)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) >= epsilon + sum([self.ensemble.weights[t]*f[(t,v.id,c)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]))#type:ignore
                else:
                    self.mdl.addConstr(sum([self.ensemble.weights[t]*f[(t,v.id,klass)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) >= sum([self.ensemble.weights[t]*f[(t,v.id,c)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]))#type:ignore
        self.mdl.addConstr(u.sum() >= 1, "sum_constraint")
        #self.mdl.setObjective(u.sum(),sense=gp.GRB.MINIMIZE)#type: ignore
        self.mdl.setObjective(sum([u[t,tree.root.id] for t,tree in idenumerate(self.ensemble)]),sense=gp.GRB.MINIMIZE)#type: ignore
    def add(self, rows : list[dict]):
        self.dataset = self.dataset._append(rows, ignore_index=True)#type: ignore
        u = self.u_vars
        f = self.f_vars
        for c in range(self.ensemble.n_classes):
            for row in rows:
                klass=self.ensemble.klass(row,tiebreaker = comp)#type:ignore
                if comp(c,klass):
                    self.mdl.addConstr(sum([self.ensemble.weights[t]*f[(t,v.id,klass)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) >= epsilon + sum([self.ensemble.weights[t]*f[(t,v.id,c)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]))#type:ignore
                else:
                    self.mdl.addConstr(sum([self.ensemble.weights[t]*f[(t,v.id,klass)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]) >= sum([self.ensemble.weights[t]*f[(t,v.id,c)] for t,tree in idenumerate(self.ensemble) for v in tree.path(row)]))#type:ignore        
    def solve(self):
        self.mdl.optimize()
        self.u = {key: round(x.X) for key,x in self.u_vars.items()}#type:ignore
        self.f = {key: round(x.X) for key,x in self.f_vars.items()}#type:ignore


    def new_ensemble(self):
        trees=[]
        for t,tree in idenumerate(self.ensemble):
            new_tree = copy.deepcopy(tree)
            if self.u[(t,new_tree.root.id)] == 1:#type:ignore
                for v in new_tree:
                    if self.u[(t,v.id)] == 1:
                        for c in range(self.ensemble.n_classes):
                            if self.f[(t,v.id,c)] == 1:#type:ignore
                                v.children = tuple()
                                v.klass = c
                                break
                trees.append(new_tree)
        for i in range(len(trees)):
            trees[i].id = i
        self.compressed = Ensemble(self.ensemble.features,trees, self.ensemble.n_classes)#type:ignore
        return self.compressed
                
    
    def check(self,dataset = None, rate = False):
        if dataset is None:
            dataset = self.dataset
        if not rate :
            print(dataset)
            for _,row in dataset.iterrows():#type:ignore
                if self.ensemble.klass(row,tiebreaker = comp) != self.compressed.klass(row,tiebreaker = comp):#type:ignore
                    return False
            return True
        else:
            s = 0
            for _,row in dataset.iterrows():#type:ignore
                if self.ensemble.klass(row,tiebreaker = comp) == self.compressed.klass(row,tiebreaker = comp):#type:ignore
                    s += 1
            return s/len(dataset)#type:ignore


    
    def update_dataset(self, df):
        self.dataset = df

if __name__ == '__main__':
    print('test')