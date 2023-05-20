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
        self.base_mdl=gp.Model(name = 'Base_Separator') #type: ignore
        self.y = None
        self.lbd = None
        self.ksi = None
        self.mu = None
        self.z = None
        self.zeta = None
        self.mdl = None


    def build_y(self):
        self.y = self.base_mdl.addVars([(t,node.id) for t in len(self.ensemble) for node in ensemble[t]],ub=[1.0 for t in len(self.ensemble) for node in ensemble[t]],vtype=gp.GRB.CONTINOUS, name="y") #type: ignore
        self.base_mdl.addConstrs((self.y[t,tree.root.id] == 1. for t,tree in idenumerate(self.ensemble)), 'y_root_is_one')
        self.base_mdl.addConstrs((self.y[t,node.id] == self.y[t,node.left.id] + self.y[t,node.right.id] for t,tree in idenumerate(self.ensemble) for node in tree.nodes), 'y_node_is_sum_of_y_children')
    
    def build_lbd(self):
        self.lbd = self.base_mdl.addVars([(t,d) for t in len(self.ensemble) for d in range(ensemble[t].depth+1)], vtype=gp.GRB.BINARY, name="lbd") #type: ignore
        self.base_mdl.addConstrs((self.lbd[t,d] >= [self.y[t,node.left.id] for node in tree.nodes_at_depth(d)].sum() for t,tree in idenumerate(self.ensemble) for d in range(tree.depth))) #type: ignore
        self.base_mdl.addConstrs((1-self.lbd[t,d] >= [self.y[t,node.right.id] for node in tree.nodes_at_depth(d)].sum() for t,tree in idenumerate(self.ensemble) for d in range(tree.depth))) #type: ignore

    def build_ksi(self):
        binary_features_id = [feature.id for feature in self.ensemble.features if feature.isbinary()]
        self.ksi = self.base_mdl.addVars(binary_features_id,vtype=gp.GRB.BINARY,name='ksi')#type: ignore
        self.base_mdl.addConstrs((self.ksi[i] <= 1 - self.y[t,node.left.id] for i in binary_features_id for t,tree in idenumerate(self.ensemble) for node in tree.nodes_with_feature(i)), 'binary_left_condtion')#type: ignore
        self.base_mdl.addConstrs((self.ksi[i] >= self.y[t,node.right.id] for i in binary_features_id for t,tree in idenumerate(self.ensemble) for node in tree.nodes_with_feature(i)), 'binary_right_condtion')#type: ignore

    def build_mu(self):
        epsilon = 10e-3
        numerical_features = [feature for feature in self.ensemble.features if feature.isnumerical()]
        self.mu = self.base_mdl.addVars([(feature.id,j) for feature in numerical_features for j in range(len(feature.levels)+1)],ub=[1.0 for feature in numerical_features for j in range(len(feature.levels)+1)],vtype=gp.GRB.CONTINOUS, name="mu")#type:ignore
        self.base_mdl.addConstrs((self.mu[feature.id,j-1] >= self.mu[feature.id,j] for feature in numerical_features for j in range(1,len(feature.levels)+1)))
        self.base_mdl.addConstrs((self.mu[feature.id,j] <= 1 - self.y[t,node.left.id] for feature in numerical_features for t,tree in idenumerate(self.ensemble) for j in range(1,len(feature.levels)+1) for node in tree.nodes_with_feature_and_level(feature,([feature.levels[0]-1]+feature.levels)[j])))#type: ignore
        self.base_mdl.addConstrs((self.mu[feature.id,j-1] >= self.y[t,node.right.id] for feature in numerical_features for t,tree in idenumerate(self.ensemble) for j in range(1,len(feature.levels)+1) for node in tree.nodes_with_feature_and_level(feature,([feature.levels[0]-1]+feature.levels)[j])))#type: ignore
        self.base_mdl.addConstrs((self.mu[feature.id,j] >= epsilon * self.y[t,node.right.id] for feature in numerical_features for t,tree in idenumerate(self.ensemble) for j in range(1,len(feature.levels)+1) for node in tree.nodes_with_feature_and_level(feature,([feature.levels[0]-1]+feature.levels)[j])))#type: ignore

    def build_z(self):
        self.z = self.base_mdl.addVars(self.ensemble.n_classes,vtype=gp.GRB.CONTINOUS,name='z')#type: ignore
        self.base_mdl.addConstrs((self.z[c] == [self.ensemble.weights[t]*tree.p(c)[v]*self.y[t,node.id] for t,tree in idenumerate(self.ensemble) for v,node in enumerate(tree.leaves)].sum() for c in range(self.ensemble.n_classes)), 'z_definition')#type: ignore

    def build_base(self):
        self.build_y()
        self.build_lbd()
        self.build_ksi()
        self.build_mu()
        self.build_z()

    def build_zeta(self,c,g):
        self.zeta = self.base_mdl.addVars([c,g],vtype=gp.GRB.CONTINOUS,name='zeta')#type: ignore
        self.mdl.addConstrs((self.zeta[k] == [self.u[t]*self.ensemble.weights[t]*tree.p(k)[v]*self.y[t,node.id] for t,tree in idenumerate(self.ensemble) for v,node in enumerate(tree.leaves)].sum() for k in [c,g]), 'z_definition')#type: ignore

    def build_class_constraint(self,c):
        self.mdl.addConstrs((self.z[c] >= self.z[g] for g in range(self.ensemble.n_classes) if c != g),'class_constraint')#type:ignore

    def build_mdl(self,c,g): 
        assert c != g, 'The class c and g must be different'
        self.base_mdl.update()
        self.base_mdl.resetParams()
        self.mdl=self.base_mdl.copy()
        self.build_zeta(c,g)
        self.build_class_constraint(c)
        self.mdl.setObjective(self.zeta[c] - self.zeta[g], sense=gp.GRB.MINIMIZE) #type: ignore
    
    def get_X(self):
        row = {}
        for feature in self.ensemble.features:
            if feature.isbinary():
                row[feature.name] = self.ksi[feature.id].X#type: ignore
            if feature.isnumerical():
                levels=feature.levels
                extended_levels = [levels[0] - 1] + levels + [levels[-1] + 1]
                row[feature.name] = sum([(extended_levels[j+1]-extended_levels[j]) * self.mu[feature.id,j].X for j in range(len(levels)+1)])#type: ignore

    def solve(self,c,g):
        self.build_mdl(c,g)
        self.get_X()


