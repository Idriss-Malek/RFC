import pandas as pd
import numpy as np
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

def compress(
    ensemble: str | TreeEnsemble,
    dataset: str | pd.DataFrame,
):
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

if __name__ == '__main__':
    import pathlib
    root = pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve()
    dataset = root / 'resources/datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.full.csv'
    ensemble = root / 'resources/forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF8.txt'
    dataset = str(dataset)
    ensemble = str(ensemble)
    compress(ensemble, dataset)

