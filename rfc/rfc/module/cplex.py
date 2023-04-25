import numpy as np
import docplex.mp.model as cpx
import docplex.mp.dvar as cpv

from ..structs.feature import FeatureType
from ..structs.ensemble import TreeEnsemble

epsilon = 1e-10

def getU(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
) -> list[cpv.Var]:
    return mdl.binary_var_list(len(ensemble))

def setUCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    u: list[cpv.Var],
    x: np.ndarray,
    lazy: bool = False
):
    n_classes = ensemble.n_classes
    w = ensemble.weights
    F = ensemble.getF(x)
    probs = F.dot(w)
    g = np.argmax(probs)
    lhs = mdl.dot(u, F[g] * w)
    for c in range(n_classes):
        if c == g:
            continue
        rhs = mdl.dot(u, F[c] * w)
        if lazy:
            mdl.add_lazy_constraint(lhs >= rhs)
        else:
            mdl.add_constraint_(lhs >= rhs)

def setUGCons(
    mdl: cpx.Model,
    u: list[cpv.Var]
):
    mdl.add_constraint_(sum(u) >= 1)

def setUObj(
    mdl: cpx.Model,
    u: list[cpv.Var]
):
    mdl.minimize(sum(u))

def getY(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
) -> dict[tuple[int, int], cpv.Var]:
    keys = []
    for t, tree in enumerate(ensemble):
        for node in tree:
            keys.append((t, node.id))
    return mdl.continuous_var_dict(keys, lb=0.0, ub=1.0, name='y')

def getLambda(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
) -> dict[tuple[int, int], cpv.Var]:
    keys = []
    for t, tree in enumerate(ensemble):
        for d in range(tree.depth):
            keys.append((t, d))
    return mdl.binary_var_dict(keys, name='lambda')

def setYRootCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    y: dict[tuple[int, int], cpv.Var]
):
    for t, tree in enumerate(ensemble):
        mdl.add_constraint_(y[(t, tree.root.id)] == 1.0)

def setYChildCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    y: dict[tuple[int, int], cpv.Var]
):
    for t, tree in enumerate(ensemble):
        for node in tree.getNodes():
            mdl.add_constraint_(y[(t, node.id)] == y[(t, node.left.id)] + y[(t, node.right.id)])

def setYDepthCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    y: dict[tuple[int, int], cpv.Var],
    lam: dict[tuple[int, int], cpv.Var]
):
    for t, tree in enumerate(ensemble):
        for d in range(tree.depth):
            y_ = [y[(t, node.left.id)] for node in tree.getNodesAtDepth(depth=d)]
            mdl.add_constraint_(sum(y_) <= lam[(t, d)])

def getMu(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
) -> dict[tuple[int, int], cpv.Var]:
    keys = []
    for feature in ensemble.features:
        if feature.ftype == FeatureType.NUMERICAL:
            f = feature.id
            k = len(feature.levels)
            for j in range(k):
                keys.append((f, j))
    return mdl.binary_var_dict(keys, name='mu')

def setMuLevelCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    mu: dict[tuple[int, int], cpv.Var]
):
    for feature in ensemble.features:
        if feature.ftype == FeatureType.NUMERICAL:
            f = feature.id
            k = len(feature.levels)
            for j in range(1, k):
                mdl.add_constraint_(mu[(f, j-1)] >= mu[(f, j)])

def setMuNodesCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    mu: dict[tuple[int, int], cpv.Var],
    y: dict[tuple[int, int], cpv.Var],
    epsilon: float = 1e-10
):
    for feature in ensemble.features:
        if feature.ftype == FeatureType.NUMERICAL:
            k = len(feature.levels)
            f = feature.id
            for j in range(1, k):
                for t, tree in enumerate(ensemble):
                    for node in tree.getNodesWithFeature(f):
                        if 0.5 * (1 + np.tanh(node.threshold)) == feature.levels[j]:
                            mdl.add_constraint_(mu[(f, j)] <= 1 - y[(t, node.left.id)])
                            mdl.add_constraint_(mu[(f, j-1)] >= y[(t, node.right.id)])
                            mdl.add_constraint_(mu[(f, j)] <= epsilon * y[(t, node.right.id)])

def getNu(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
) -> dict[tuple[int, int], cpv.Var]:
    keys = []
    for feature in ensemble.features:
        if feature.ftype == FeatureType.CATEGORICAL:
            f = feature.id
            categories = feature.categories
            for c in categories:
                keys.append((f, c))
    return mdl.binary_var_dict(keys, name='nu')

def setNuNodesCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    nu: dict[tuple[int, int], cpv.Var],
    y: dict[tuple[int, int], cpv.Var],
):
    for feature in ensemble.features:
        if feature.ftype == FeatureType.CATEGORICAL:
            f = feature.id
            categories = feature.categories
            for c in categories:
                for t, tree in enumerate(ensemble):
                    for node in tree.getNodesWithFeature(f):
                        if c in node.categories: # TODO: adapt to categorical values.
                            mdl.add_constraint_(nu[(f, c)] <= 1 - y[(t, node.left.id)])
                            mdl.add_constraint_(nu[(f, c)] >= y[(t, node.right.id)])

def getX(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
) -> dict[int, cpv.Var]:
    keys = []
    for feature in ensemble.features:
        if feature.ftype == FeatureType.BINARY:
            keys.append(feature.id)
    return mdl.binary_var_dict(keys, name='x')

def setXBinaryCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    x: dict[int, cpv.Var],
    y: dict[tuple[int, int], cpv.Var],
):
    for feature in ensemble.features:
        if feature.ftype == FeatureType.BINARY:
            f = feature.id
            for t, tree in enumerate(ensemble):
                for node in tree.getNodesWithFeature(f):
                    mdl.add_constraint_(x[f] <= 1 - y[(t, node.left.id)])
                    mdl.add_constraint_(x[f] >= y[(t, node.right.id)])

def getZ(
    mdl: cpx.Model,
    ensemble: TreeEnsemble
) -> list[cpv.Var]:
    return mdl.binary_var_list(ensemble.n_classes)

def setZDefCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    z: list[cpv.Var],
    y: dict[tuple[int, int], cpv.Var]
):
    w = ensemble.weights
    for c in range(ensemble.n_classes):
        s = []
        for t, tree in enumerate(ensemble):
            p = tree.getProbas(c)
            yl = [y[(t, node.id)] for node in tree.getLeaves()]
            s.append(mdl.dot(p, yl))
        mdl.add_constraint_(z[c] == mdl.dot(w, s))

def setZKlassCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    z: list[cpv.Var],
    c: int  
):
    for g in range(ensemble.n_classes):
        if g != c:
            mdl.add_constraint_(z[c] >= z[g])

def getZeta(
    mdl: cpx.Model
) -> list[cpv.Var]:
    return mdl.binary_var_list(2)

def setZetaCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    zeta: list[cpv.Var],
    y: dict[tuple[int, int], cpv.Var],
    u: np.ndarray,
    c: int,
    cc: int
):
    w = ensemble.weights
    wu = w * u
    s = []
    for t, tree in enumerate(ensemble):
        p = tree.getProbas(c)
        yl = [y[(t, node.id)] for node in tree.getLeaves()]
        s.append(mdl.dot(p, yl))
    mdl.add_constraint_(zeta[0] == mdl.dot(wu, s))

    s = []
    for t, tree in enumerate(ensemble):
        p = tree.getProbas(cc)
        yl = [y[(t, node.id)] for node in tree.getLeaves()]
        s.append(mdl.dot(p, yl))
    mdl.add_constraint_(zeta[1] == mdl.dot(wu, s))

def setZetaObj(
    mdl: cpx.Model,
    zeta: list[cpv.Var]
):
    mdl.minimize(zeta[0] - zeta[1])