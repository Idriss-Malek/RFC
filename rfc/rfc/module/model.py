import numpy as np
import docplex.mp.model as cpx
import docplex.mp.dvar as cpv

from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr

from ..types import *
from ..structs.feature import FeatureType
from ..structs.ensemble import Ensemble
from ..structs.tree import Tree
from ..structs.utils import idenumerate

epsilon = 1e-10

class Model(cpx.Model):
    __built: bool = False 
    
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(name='Compression', **kwargs)

    def prob(
        self,
        E: Ensemble,
        x: Sample,
        c: int,
        u: dict[int, Var]
    ):
        w = E.w()
        F = E.F(x)
        return sum(w[t] * F[t][c] * u[t] for t, _ in idenumerate(E))

    def build(
        self,
        E: Ensemble,
        on: Dataset,
        lazy: bool = False
    ):
        # The variables
        u: dict[int, Var]
        
        # For each tree, create a variable
        keys = list(t for t, _ in idenumerate(E))
        u = self.binary_var_dict(keys, name='u')

        # For each observation, add majority class constraints
        for _, row in on.iterrows():
            
            # Get the observation.
            x = np.array(row.values)
            self.__add_sample(E, x, u, lazy)
        
        # Add the constraint that at least one tree must be used.
        self.add_constraint_(sum(u) >= 1)

        # Minimize the sum of the weights.
        self.minimize(sum(u))

        # The model has been built.
        self.__built = True

        return self

    def add_cut(self, E: Ensemble, x: Sample, lazy: bool = False):
        if not self.__built:
            raise Exception('Model has not built yet.')
        u = self.__get_u(E)
        self.__add_sample(E, x, u, lazy)

    def __add_sample(self, E: Ensemble, x: Sample, u: dict[int, Var], lazy: bool = False):
        # Get the majority class of x.
        g = E.klass(x)

        # Add the majority class constraints.
        lhs = self.prob(E, x, g, u)
        for c in range(E.n_classes):
            if c == g: continue
            rhs = self.prob(E, x, c, u)
            if lazy: self.add_lazy_constraint(lhs >= rhs)
            else: self.add_constraint_(lhs >= rhs)

    def __get_u(self, E: Ensemble) -> dict[int, Var]:
        _u = self.find_matching_vars('u')
        return {t: _u[t] for t, _ in idenumerate(E)}

class CounterFactual(cpx.Model):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__('CounterFactual', **kwargs)

    def prob(
        self,
        E: Ensemble,
        c: int,
        y: dict[tuple[int, int], Var],
        u: None | list[float] | dict[int, float] = None
    ):
        w = E.w(u)
        return self.sum(w[t] * L.p(c) * y[(t, v)] for t, T in idenumerate(E) for v, L in idenumerate(T.leaves))

    def build(
        self,
        E: Ensemble,
        u: list[float] | dict[int, float],
        c: int,
        g: int
    ):
        y: dict[tuple[int, int], Var]
        lam: dict[tuple[int, int], Var]
        xi: dict[int, Var]
        mu: dict[tuple[int, int], Var]
        z: dict[int, Var]
        zeta: dict[int, Var]

        keys = []
        for t, T in idenumerate(E):
            for v, N in idenumerate(T.nodes):
                keys.append((t, v))
        y = self.continuous_var_dict(keys, lb=0.0, ub=1.0, name='y')

        keys = []
        for t, T in idenumerate(E):
            for d in range(T.depth):
                keys.append((t, d))
        lam = self.binary_var_dict(keys, name='lambda')

        for t, T in idenumerate(E):
            v = T.root.id
            self.add_constraint_(y[(t, v)] == 1.0)

        for t, T in idenumerate(E):
            for v, N in idenumerate(T.nodes):
                l, r = N.left.id, N.right.id
                self.add_constraint_(y[(t, v)] == y[(t, l)] + y[(t, r)])
        
        for t, T in idenumerate(E):
            for d in range(T.depth):
                yl, yr = [], []
                for N in T.nodes_at_depth(d):
                    l, r = N.left.id, N.right.id
                    yl.append(y[(t, l)])
                    yr.append(y[(t, r)])
                self.add_constraint_(sum(yl) <= lam[(t, d)])
                self.add_constraint_(sum(yr) <= 1-lam[(t, d)])

        keys = [f for f, _  in idenumerate(E.binary_features)]
        xi = self.binary_var_dict(keys, name='xi')

        for f, F in idenumerate(E.binary_features):
            for t, T in idenumerate(E):
                for N in T.nodes_with_feature(f):
                    l, r = N.left.id, N.right.id
                    self.add_constraint_(xi[f] >= 1 - y[(t, l)])
                    self.add_constraint_(xi[f] >= y[(t, r)])

        keys = []
        for f, F in idenumerate(E.numerical_features):
            k = len(F.levels)
            for j in range(k+1):
                keys.append((f, j))
        mu = self.continuous_var_dict(keys, lb=0.0, ub=1.0, name='mu')

        for f, F in idenumerate(E.numerical_features):
            k = len(F.levels)
            for j in range(k):
                self.add_constraint_(mu[(f, j)] >= mu[(f, j+1)])

        for f, F in idenumerate(E.numerical_features):
            k = len(F.levels)
            for j in range(k):
                for t, T in idenumerate(E):
                    for N in T.nodes_with_feature(F):
                        if F.levels[j] == N.threshold:
                            l, r = N.left.id, N.right.id
                            self.add_constraint_(mu[(f, j+1)] <= 1 - y[(t, l)])
                            self.add_constraint_(mu[(f, j)] >= y[(t, r)])
                            self.add_constraint_(mu[(f, j+1)] <= epsilon * y[(t, r)])

        keys = list(range(E.n_classes))
        z = self.continuous_var_dict(keys, name='z')

        for cc in range(E.n_classes):
            self.add_constraint_(z[cc] == self.prob(E, cc, y))

        for cc in range(E.n_classes):
            if cc != c: self.add_constraint_(z[c] >= z[cc])

        keys = [c, g]
        zeta = self.continuous_var_dict(keys, name='zeta')

        self.add_constraint_(zeta[c] == self.prob(E, c, y, u))
        self.add_constraint_(zeta[g] == self.prob(E, g, y, u))
        self.minimize(z[c] - z[g])

        return self



def getU(
    mdl: cpx.Model,
    ensemble: Ensemble
) -> list[Var]:
    return mdl.binary_var_list(len(ensemble))

def setUCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    u: list[Var],
    x: np.ndarray,
    lazy: bool = False
):
    n_classes = ensemble.n_classes
    w = ensemble.weights
    F = ensemble.F(x)
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
    u: list[Var]
):
    mdl.add_constraint_(sum(u) >= 1)

def setUObj(
    mdl: cpx.Model,
    u: list[Var]
):
    mdl.minimize(sum(u))

def getY(
    mdl: cpx.Model,
    ensemble: Ensemble
) -> dict[tuple[int, int], Var]:
    keys = []
    for t, tree in enumerate(ensemble):
        for node in tree:
            keys.append((t, node.id))
    return mdl.continuous_var_dict(keys, lb=0.0, ub=1.0, name='y')

def getLambda(
    mdl: cpx.Model,
    ensemble: Ensemble
) -> dict[tuple[int, int], Var]:
    keys = []
    for t, tree in enumerate(ensemble):
        for d in range(tree.depth):
            keys.append((t, d))
    return mdl.binary_var_dict(keys, name='lambda')

def setYRootCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    y: dict[tuple[int, int], Var]
):
    for t, tree in enumerate(ensemble):
        mdl.add_constraint_(y[(t, tree.root.id)] == 1.0)

def setYChildCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    y: dict[tuple[int, int], Var]
):
    for t, tree in enumerate(ensemble):
        for node in tree.nodes:
            mdl.add_constraint_(y[(t, node.id)] == y[(t, node.left.id)] + y[(t, node.right.id)])

def setYDepthCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    y: dict[tuple[int, int], Var],
    lam: dict[tuple[int, int], Var]
):
    for t, tree in enumerate(ensemble):
        for d in range(tree.depth):
            y_ = [y[(t, node.left.id)] for node in tree.nodes_at_depth(depth=d)]
            mdl.add_constraint_(sum(y_) <= lam[(t, d)])
            y_ = [y[(t, node.right.id)] for node in tree.nodes_at_depth(depth=d)]
            mdl.add_constraint_(sum(y_) <= 1 - lam[(t, d)])
            

def getMu(
    mdl: cpx.Model,
    ensemble: Ensemble
) -> dict[tuple[int, int], Var]:
    keys = []
    for feature in ensemble.features:
        if feature.type == FeatureType.NUMERICAL:
            f = feature.id
            k = len(feature.levels) + 1
            for j in range(k):
                keys.append((f, j))
    return mdl.continuous_var_dict(keys, lb=0.0, ub=1.0, name='mu')

def setMuLevelCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    mu: dict[tuple[int, int], Var]
):
    for feature in ensemble.features:
        if feature.type == FeatureType.NUMERICAL:
            f = feature.id
            k = len(feature.levels) + 1
            for j in range(1, k):
                mdl.add_constraint_(mu[(f, j-1)] >= mu[(f, j)])

def setMuNodesCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    mu: dict[tuple[int, int], Var],
    y: dict[tuple[int, int], Var],
    epsilon: float = 1e-10
):
    for feature in ensemble.features:
        if feature.type == FeatureType.NUMERICAL:
            f = feature.id
            k = len(feature.levels) + 1
            for j in range(1, k):
                for t, tree in enumerate(ensemble):
                    for node in tree.nodes_with_feature(f):
                        if node.threshold == feature.levels[j-1]:
                            mdl.add_constraint_(mu[(f, j)] <= 1 - y[(t, node.left.id)])
                            mdl.add_constraint_(mu[(f, j-1)] >= y[(t, node.right.id)])
                            mdl.add_constraint_(mu[(f, j)] >= epsilon * y[(t, node.right.id)])

def getNu(
    mdl: cpx.Model,
    ensemble: Ensemble
) -> dict[tuple[int, int], Var]:
    keys = []
    for feature in ensemble.features:
        if feature.type == FeatureType.CATEGORICAL:
            f = feature.id
            categories = feature.categories
            for c in categories:
                keys.append((f, c))
    return mdl.binary_var_dict(keys, name='nu')

def setNuNodesCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    nu: dict[tuple[int, int], Var],
    y: dict[tuple[int, int], Var],
):
    for feature in ensemble.features:
        if feature.type == FeatureType.CATEGORICAL:
            f = feature.id
            categories = feature.categories
            for c in categories:
                for t, tree in enumerate(ensemble):
                    for node in tree.nodes_with_feature(f):
                        if c in node.categories: # TODO: adapt to categorical values.
                            mdl.add_constraint_(nu[(f, c)] <= 1 - y[(t, node.left.id)])
                            mdl.add_constraint_(nu[(f, c)] >= y[(t, node.right.id)])

def getXi(
    mdl: cpx.Model,
    ensemble: Ensemble
) -> dict[int, Var]:
    keys = []
    for feature in ensemble.features:
        if feature.type == FeatureType.BINARY:
            keys.append(feature.id)
    return mdl.binary_var_dict(keys, name='xi')

def setXiBinaryCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    xi: dict[int, Var],
    y: dict[tuple[int, int], Var],
):
    for feature in ensemble.features:
        if feature.type == FeatureType.BINARY:
            f = feature.id
            for t, tree in enumerate(ensemble):
                for node in tree.nodes_with_feature(f):
                    mdl.add_constraint_(xi[f] <= 1 - y[(t, node.left.id)])
                    mdl.add_constraint_(xi[f] >= y[(t, node.right.id)])

def getZ(
    mdl: cpx.Model,
    ensemble: Ensemble
) -> list[Var]:
    return mdl.continuous_var_list(ensemble.n_classes)

def getKlassProb(
    mdl: cpx.Model,
    tree: Tree,
    y: dict[tuple[int, int], Var],
    c: int
):
    p = tree.p(c)
    yl = [y[(tree.id, leaf.id)] for leaf in tree.leaves]
    return mdl.dot(yl, p)

def getKlassProbs(
    mdl: cpx.Model,
    ensemble: Ensemble,
    y: dict[tuple[int, int], Var],
    c: int
):
    return [getKlassProb(mdl, tree, y, c) for tree in ensemble]

def setZDefCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    z: list[Var],
    y: dict[tuple[int, int], Var]
):
    w = ensemble.weights
    for c in range(ensemble.n_classes):
        z[c].set_name(f'z_{c}')
        s = getKlassProbs(mdl, ensemble, y, c)
        mdl.add_constraint_(z[c] == mdl.dot(s, w))

def setZKlassCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    z: list[Var],
    c: int  
):
    for g in range(ensemble.n_classes):
        if g != c:
            mdl.add_constraint_(z[c] >= z[g])

def getZeta(
    mdl: cpx.Model
) -> list[Var]:
    return mdl.continuous_var_list(2)

def setZetaCons(
    mdl: cpx.Model,
    ensemble: Ensemble,
    zeta: list[Var],
    y: dict[tuple[int, int], Var],
    u: np.ndarray,
    c: int,
    g: int
):
    w = ensemble.weights
    wu = w * u
    zeta[0].set_name(f'zeta_{c}')
    s = getKlassProbs(mdl, ensemble, y, c)
    mdl.add_constraint_(zeta[0] == mdl.dot(s, wu))

    zeta[1].set_name(f'zeta_{g}')
    s = getKlassProbs(mdl, ensemble, y, g)
    mdl.add_constraint_(zeta[1] == mdl.dot(s, wu))

def setZetaObj(
    mdl: cpx.Model,
    zeta: list[Var]
):
    mdl.minimize(zeta[0] - zeta[1])