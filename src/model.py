import pandas as pd
import numpy as np
import docplex.mp.model as cpx

from anytree import PreOrderIter
from tree import *

epsilon = 1e-10

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
    keys = []
    for t, tree in enumerate(ensemble.trees):
        for node in PreOrderIter(tree.root):
            keys.append((t, node.name))
    mdl.var_dict(keys, lb=0, ub=1, name='y') # type: ignore

def addLambda(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    keys = []
    for t, tree in enumerate(ensemble.trees):
        md = tree.root.height
        for d in range(md):
            keys.append((t, d))
    mdl.binary_var_dict(keys, name='lambda') # type: ignore

def addMu(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    pass

def addNu(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    pass

def addX(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    pass

def addZ(mdl: cpx.Model, ensemble: TreeEnsemble):
    pass

def addZeta(mdl: cpx.Model, ensemble: TreeEnsemble):
    pass

def addBaseCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    y = mdl.find_matching_vars('y')
    lam = mdl.find_matching_vars('lambda')
    for t, tree in enumerate(ensemble.trees):
        mdl.add_constraint_(y[(t, tree.root.name)] == 1) # type: ignore
        for node in PreOrderIter(tree.root):
            if not node.is_leaf:
                left = node.children[0]
                right = node.children[1]
                lhs = y[(t, node.name)] # type: ignore
                rhs = y[(t, left.name)] + y[(t, right.name)] # type: ignore
                mdl.add_constraint_(lhs == rhs)
        md = tree.root.height
        for d in range(md):
            nodes = []
            for node in PreOrderIter(tree.root):
                if node.depth == d and not node.is_leaf:
                    left = node.children[0]
                    nodes.append(left)
            lhs = mdl.sum(y[(t, node.name)] for node in nodes) # type: ignore
            rhs = lam[(t, d)] # type: ignore
            mdl.add_constraint_(lhs <= rhs)

def addMuCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
    pass

def addXCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
):
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
    addY(mdl, ensemble)
    addLambda(mdl, ensemble)
    addBaseCons(mdl, ensemble)
    addMu(mdl, ensemble)
    addMuCons(mdl, ensemble)
    # addNu(mdl, ensemble)
    addX(mdl, ensemble)
    addXCons(mdl, ensemble)

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