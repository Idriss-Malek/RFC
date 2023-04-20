import pandas as pd
import numpy as np
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
    F = ensemble.getF(x)
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

def getSolution(mdl: cpx.Model):
    u = mdl.find_matching_vars('u')
    n_trees = len(u)
    v = np.empty(n_trees)
    for t in range(n_trees):
        v[t] = u[t].solution_value
    return v

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

def getZeta(mdl: cpx.Model):
    pass

def getX(mdl: cpx.Model):
    pass

def buildSeparationModel(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    u: np.ndarray
):
    pass

def separate(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    **kwargs
):
    log_output = kwargs.get('log_output', False)
    if not isinstance(log_output, bool):
        raise TypeError('log_output must be a boolean')
    
    precision = kwargs.get('precision', 8)
    if not isinstance(precision, int):
        raise TypeError('precision must be an integer')

    sols = []
    with cpx.Model(
        name="Separation",
        log_output=log_output,
        float_precision=precision
    ) as mdl:
        buildSeparationModel(mdl, ensemble, u)
        for c in range(ensemble.n_classes):
            for cc in range(ensemble.n_classes):
                if cc == c:
                    continue
                sol = mdl.solve()
                if sol:
                    zeta = getZeta(mdl)
                    if zeta[0] - zeta[1] > 0:
                        sols.append(getX(mdl))
                        pass
                    else:
                        pass
                else:
                    pass
    return sols