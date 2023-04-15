import cplex
import pandas as pd
import numpy as np
from anytree import Node
from typing import Any
from tree_ensemble_function import tree_ensemble_fun

def compress(
    trees,
    train: pd.DataFrame,
    nb_classes:int,
    weights: None | Any = None,
    on: str ='train'
):
    nb_trees=len(trees)
    if weights is None:
        weights = np.ones(len(trees))
    model = cplex.Cplex()
    u = model.variables.add(names=[f"u{t}" for t in range(nb_trees)],types=[model.variables.type.binary for t in range(nb_trees)])
    constraints = []
    rhs=[]
    senses=[]
    for index, row in train.iterrows():
        results=np.empty([nb_classes,nb_trees])
        probs=np.empty(nb_classes)
        for c in range (nb_classes):
            results[c]=tree_ensemble_fun(trees,row,c)
        probs=results.mean(axis=1)
        original_rf_class=np.argmax(probs)
        if nb_classes == 2:
            constraints.append([[u[t] for t in range(nb_trees)],
                                    [weights[t]*(results[original_rf_class,t] -results[1-original_rf_class,t]) for t in range(nb_trees)]])
            senses.append('G')
            rhs.append(0.)
        else:
            for c in range(nb_classes):
                if c!=original_rf_class:
                    constraints.append([[u[t] for t in range(nb_trees)],
                                    [weights[t]*(results[original_rf_class,t] -results[c,t]) for t in range(nb_trees)]])
                    senses.append('G')
                    rhs.append(0.)
    constraints.append([[u[t] for t in range(nb_trees)],[1 for t in range(nb_trees)]])
    senses.append('G')
    rhs.append(1.)
    model.linear_constraints.add(lin_expr=constraints,senses = senses,rhs = rhs)
    model.objective.set_sense(model.objective.sense.minimize)
    model.objective.set_linear([(u[t], 1.0) for t in range(nb_trees)])
    model.solve()
    return model.solution.get_values()


