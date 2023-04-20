import pandas as pd
import numpy as np
<<<<<<< HEAD:src/rfc/compresser.py
import docplex.mp as mp
import docplex.mp.model as cpx

from tree import *

def addVariables(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    n_trees = len(ensemble.trees)
    mdl.binary_var_list(n_trees, name='u')

def setConstraints(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    x: pd.Series
) -> None:
    n_trees = len(ensemble.trees)
    n_classes = ensemble.n_classes
    w = ensemble.weigths
    u = mdl.find_matching_vars('u')
    F = np.empty((n_classes, n_trees))
    for c in range(n_classes):
        for t in range(n_trees):
            F[c][t] = ensemble.trees[t].getF(x, c)
    probs = F.dot(w)
    gamma = np.argmax(probs)
    lhs = mdl.sum(w[t] * u[t] * F[gamma][t] for t in range(n_trees))
    for c in range(n_classes):
        if c == gamma:
            continue
        rhs = mdl.sum(w[t] * u[t] * F[c][t] for t in range(n_trees))
        mdl.add_constraint_(lhs >= rhs)

def setCover(mdl: cpx.Model):
    u = mdl.find_matching_vars('u')
    mdl.add_constraint_(mdl.sum(u) >= 1)

def setObjective(mdl: cpx.Model):
    u = mdl.find_matching_vars('u')
    mdl.minimize(mdl.sum(u))

def buildModel(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    train: pd.DataFrame,
):
    addVariables(mdl, ensemble)
    for _, x in train.iterrows():
        setConstraints(mdl, ensemble, x)
    setCover(mdl)
    setObjective(mdl)
=======
from anytree import Node
from typing import Any
from tree_ensemble_function import tree_ensemble_fun
import time
>>>>>>> bc16e12e64be01fadcecc382e4d8364036bfd97a:src/compresser.py

def compress(
    ensemble: str | TreeEnsemble,
    dataset: str | pd.DataFrame,
):
<<<<<<< HEAD:src/rfc/compresser.py
    if isinstance(ensemble, str):
        ensemble = TreeEnsemble.from_file(ensemble)
    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)
    with cpx.Model(name="RF Compresser", log_output=True, float_precision=6) as mdl:
        buildModel(mdl, ensemble, dataset)
        sol = mdl.solve()
        if sol:
            u = mdl.find_matching_vars('u')
            for ut in u:
                print("u{} = {}".format(ut.index, ut.solution_value))
        else:
            print("No solution found")
=======
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
    t1=time.perf_counter_ns()
    model.solve()
    t2=time.perf_counter_ns()-t1
    return model.solution.get_values(),t2
>>>>>>> bc16e12e64be01fadcecc382e4d8364036bfd97a:src/compresser.py

if __name__ == '__main__':
    import pathlib
    root = pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    dataset = root / 'resources/datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.full.csv'
    ensemble = root / 'resources/forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF8.txt'
    dataset = str(dataset)
    ensemble = str(ensemble)
    compress(ensemble, dataset)

