import pandas as pd
import numpy as np
import docplex.mp.model as cpx

from tree import *

def addU(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    n_trees = len(ensemble.trees)
    mdl.binary_var_list(n_trees, name='u') # type: ignore

def addCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    x: pd.Series
):
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
        mdl.add_constraint_(lhs >= rhs) # type: ignore

def setG(mdl: cpx.Model):
    u = mdl.find_matching_vars('u')
    mdl.add_constraint_(mdl.sum(u) >= 1) # type: ignore

def setObj(mdl: cpx.Model):
    u = mdl.find_matching_vars('u')
    mdl.minimize(mdl.sum(u))

def getU(mdl: cpx.Model):
    u = mdl.find_matching_vars('u')
    n_trees = len(u)
    v = np.empty(n_trees)
    for t in range(n_trees):
        v[t] = u[t].solution_value
    return v

def buildModel(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    dataset: pd.DataFrame,
):
    addU(mdl, ensemble)
    for _, x in dataset.iterrows():
        addCons(mdl, ensemble, x)
    setG(mdl)
    setObj(mdl)

def addY(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    pass

def addLambda(mdl: cpx.Model, ensemble: TreeEnsemble):
    pass

def addX(mdl: cpx.Model, ensemble: TreeEnsemble):
    pass

def addZ(mdl: cpx.Model, ensemble: TreeEnsemble):
    pass

def addZeta(mdl: cpx.Model, ensemble: TreeEnsemble):
    pass

def getZeta(mdl: cpx.Model):
    zeta = mdl.find_matching_vars('zeta')
    z = np.empty(2)
    for i in range(2):
        z[i] = zeta[i].solution_value
    return z

def getX(mdl: cpx.Model):
    pass

def buildBaseSepModel(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    pass

def buildSepModel(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    u: np.ndarray,
    c: int,
    cc: int
):
    pass

def separate(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    baseSep: None | cpx.Model = None,
    **kwargs
):
    log_output = kwargs.get('log_output', False)
    if not isinstance(log_output, bool):
        raise TypeError('log_output must be a boolean')
    
    precision = kwargs.get('precision', 8)
    if not isinstance(precision, int):
        raise TypeError('precision must be an integer')

    sols = []
    if baseSep is None:
        mdl = cpx.Model(
            name="Separation",
            log_output=log_output,
            float_precision=precision
        )
        buildBaseSepModel(mdl, ensemble)
    else:
        mdl = baseSep.clone()

    for c in range(ensemble.n_classes):
        for cc in range(ensemble.n_classes):
            if cc == c:
                continue
            buildSepModel(mdl, ensemble, u, c, cc)
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