import pandas as pd
import numpy as np

import gurobipy as gp

from ..structs.ensemble import Ensemble
from ..structs.utils import idenumerate

class Separator:
    ensemble : Ensemble
    lazy : bool
    def __init__(
        self,
        ensemble: Ensemble,
        u : list[float] | dict,
        lazy: bool = False
    ) -> None:
        if not isinstance(ensemble, Ensemble):
            raise TypeError('ensemble must be a TreeEnsemble')
        if not isinstance(lazy, bool):
            raise TypeError('lazy must be a boolean')
        self.ensemble = ensemble
        self.lazy = lazy
        self.u = u
        self.sep = []
        self.mdl = None #type: ignore
        self.y = None
        self.lbd = None
        self.ksi = None
        self.mu = None
        self.z = None
        self.zeta = None
    
    def update_u(self,u):
        self.u = u

    def build_y(self):
        self.y = self.mdl.addVars([(t,node.id) for t,tree in idenumerate(self.ensemble) for node in tree],ub=[1.0 for t,tree in idenumerate(self.ensemble) for node in tree],vtype=gp.GRB.CONTINUOUS, name="y") #type: ignore
        self.mdl.addConstrs((self.y[t,tree.root.id] == 1. for t,tree in idenumerate(self.ensemble)))#type: ignore
        self.mdl.addConstrs((self.y[t,node.id] == self.y[t,node.left.id] + self.y[t,node.right.id] for t,tree in idenumerate(self.ensemble) for node in tree.nodes))#type: ignore
    
    def build_lbd(self):
        self.lbd = self.mdl.addVars([(t,d) for t,tree in idenumerate(self.ensemble) for d in range(tree.depth+1)], vtype=gp.GRB.BINARY, name="lbd") #type: ignore
        self.mdl.addConstrs((self.lbd[t,d] >= sum([self.y[t,node.left.id] for node in tree.nodes_at_depth(d)]) for t,tree in idenumerate(self.ensemble) for d in range(tree.depth))) #type: ignore
        self.mdl.addConstrs((1-self.lbd[t,d] >= sum([self.y[t,node.right.id] for node in tree.nodes_at_depth(d)]) for t,tree in idenumerate(self.ensemble) for d in range(tree.depth))) #type: ignore

    def build_ksi(self):
        binary_features_id = [feature.id for feature in self.ensemble.features if feature.isbinary()]
        self.ksi = self.mdl.addVars(binary_features_id,vtype=gp.GRB.BINARY,name='ksi')#type: ignore
        self.mdl.addConstrs((self.ksi[i] <= 1 - self.y[t,node.left.id] for i in binary_features_id for t,tree in idenumerate(self.ensemble) for node in tree.nodes_with_feature(i)))#type: ignore
        self.mdl.addConstrs((self.ksi[i] >= self.y[t,node.right.id] for i in binary_features_id for t,tree in idenumerate(self.ensemble) for node in tree.nodes_with_feature(i)))#type: ignore

    def build_mu(self):
        epsilon = 10e-3
        numerical_features = [feature for feature in self.ensemble.features if feature.isnumerical()]
        self.mu = self.mdl.addVars([(feature.id,j) for feature in numerical_features for j in range(len(feature.levels)+1)],ub=[1.0 for feature in numerical_features for j in range(len(feature.levels)+1)],vtype=gp.GRB.CONTINUOUS, name="mu")#type:ignore
        self.mdl.addConstrs((self.mu[feature.id,j-1] >= self.mu[feature.id,j] for feature in numerical_features for j in range(1,len(feature.levels)+1)))#type: ignore
        self.mdl.addConstrs((self.mu[feature.id,j] <= 1 - self.y[t,node.left.id] for feature in numerical_features for t,tree in idenumerate(self.ensemble) for j in range(1,len(feature.levels)+1) for node in tree.nodes_with_feature_and_level(feature,([feature.levels[0]-1]+feature.levels)[j])))#type: ignore
        self.mdl.addConstrs((self.mu[feature.id,j-1] >= self.y[t,node.right.id] for feature in numerical_features for t,tree in idenumerate(self.ensemble) for j in range(1,len(feature.levels)+1) for node in tree.nodes_with_feature_and_level(feature,([feature.levels[0]-1]+feature.levels)[j])))#type: ignore
        self.mdl.addConstrs((self.mu[feature.id,j] >= epsilon * self.y[t,node.right.id] for feature in numerical_features for t,tree in idenumerate(self.ensemble) for j in range(1,len(feature.levels)+1) for node in tree.nodes_with_feature_and_level(feature,([feature.levels[0]-1]+feature.levels)[j])))#type: ignore

    def build_z(self):
        self.z = self.mdl.addVars(self.ensemble.n_classes,vtype=gp.GRB.CONTINUOUS,name='z')#type: ignore
        self.mdl.addConstrs((self.z[c] == sum([self.ensemble.weights[t]*tree.p(c)[v]*self.y[t,node.id] for t,tree in idenumerate(self.ensemble) for v,node in enumerate(tree.leaves)]) for c in range(self.ensemble.n_classes)), 'z_definition')#type: ignore

    def build_base(self):
        self.build_y()
        self.build_lbd()
        self.build_ksi()
        self.build_mu()
        self.build_z()

    def build_zeta(self,c,g):
        self.zeta = self.mdl.addVars([c,g],vtype=gp.GRB.CONTINUOUS,name='zeta')#type: ignore
        self.mdl.addConstrs((self.zeta[k] == sum([self.u[t]*self.ensemble.weights[t]*tree.p(k)[v]*self.y[t,node.id] for t,tree in idenumerate(self.ensemble) for v,node in enumerate(tree.leaves)]) for k in [c,g]), 'zeta_definition')#type: ignore

    def build_class_constraint(self,c):
        self.mdl.addConstrs((self.z[c] >= 0.01 + self.z[g] for g in range(self.ensemble.n_classes) if c != g),'class_constraint')#type:ignore

    def build_mdl(self,c,g): 
        self.mdl = gp.Model(name = f'Separator_{c}_{g}')#type: ignore
        self.build_base()
        self.build_zeta(c,g)
        self.build_class_constraint(c)
        self.mdl.setObjective(self.zeta[c] - self.zeta[g], sense=gp.GRB.MINIMIZE) #type: ignore
    
    def get_X(self):
        row = {}
        for feature in self.ensemble.features:
            if feature.isbinary():
                row[feature.id] = self.ksi[feature.id].X#type: ignore
            if feature.isnumerical():
                levels=feature.levels
                extended_levels = [0] + levels + [levels[-1] + 1]
                row[feature.id] = sum([(extended_levels[j+1]-extended_levels[j]) * self.mu[feature.id,j].X for j in range(len(levels)+1)])#type: ignore
        return row

    def solve(self,c,g):
        self.build_mdl(c,g)
        self.mdl.optimize()#type:ignore
        if self.mdl.getObjective().getValue() < 0:#type:ignore
            return self.get_X()
        else:
            return
    
    def find_all(self):
        rows=[]
        for c in range(self.ensemble.n_classes):
            for g in range(self.ensemble.n_classes):
                if c != g:
                    row=self.solve(c,g)
                    if row :
                        print(c,g)
                        print(self.ensemble.klass(row))
                        print(self.ensemble.klass(row,self.u))
                        if self.ensemble.klass(row) != self.ensemble.klass(row,self.u):
                            rows.append(row)
        return rows



