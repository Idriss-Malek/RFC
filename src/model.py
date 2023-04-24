import pandas as pd
import numpy as np
import docplex.mp.model as cpx
import docplex.mp.dvar as cpv

from tree import *

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
    x: np.ndarray
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
        for node in tree.getNodes(leaves=False):
            mdl.add_constraint_(y[(t, node.id)] == y[(t, node.left.id)] + y[(t, node.right.id)])

def setYDepthCons(
    mdl: cpx.Model,
    ensemble: TreeEnsemble,
    y: dict[tuple[int, int], cpv.Var],
    lam: dict[tuple[int, int], cpv.Var]
):
    for t, tree in enumerate(ensemble):
        for d in range(tree.depth):
            y_ = [y[(t, node.left.id)] for node in tree.getNodesAtDepth(depth=d, leaves=False)]
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
            for j in range(k):
                for t, tree in enumerate(ensemble):
                    for node in tree.getNodesWithFeature(f, leaves=False):
                        if node.threshold == feature.levels[j]:
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
                    for node in tree.getNodesWithFeature(f, leaves=False):
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
                for node in tree.getNodesWithFeature(f, leaves=False):
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
    mdl.maximize(zeta[0] - zeta[1])

class TreeEnsembleSeparator:
    ensemble: TreeEnsemble
    mdl: cpx.Model
    y: dict[tuple[int, int], cpv.Var]
    lam: dict[tuple[int, int], cpv.Var]
    x: dict[int, cpv.Var]
    z: list[cpv.Var]
    zeta: list[cpv.Var]
    mu: dict[tuple[int, int], cpv.Var]
    nu: dict[tuple[int, int], cpv.Var]
    epsilon: float
    
    def __init__(
        self,
        ensemble: TreeEnsemble,
        epsilon: float = 1e-10  
    ) -> None:
        self.ensemble = ensemble
        self.epsilon = epsilon

    def addY(self):
        self.y = getY(self.mdl, self.ensemble)

    def addLambda(self):
        self.lam = getLambda(self.mdl, self.ensemble)

    def addYCons(self):
        setYRootCons(self.mdl, self.ensemble, self.y)
        setYChildCons(self.mdl, self.ensemble, self.y)
        setYDepthCons(self.mdl, self.ensemble, self.y, self.lam)

    def addMu(self):
        self.mu = getMu(self.mdl, self.ensemble)

    def addMuCons(self):
        setMuLevelCons(self.mdl, self.ensemble, self.mu)
        setMuNodesCons(self.mdl, self.ensemble, self.mu, self.y, self.epsilon)      

    def addNu(self):
        self.nu = getNu(self.mdl, self.ensemble)

    def addNuCons(self):
        setNuNodesCons(self.mdl, self.ensemble, self.nu, self.y)

    def addX(self):
        self.x = getX(self.mdl, self.ensemble)

    def addXCons(self):
        setXBinaryCons(self.mdl, self.ensemble, self.x, self.y)

    def addZ(self):
        self.z = getZ(self.mdl, self.ensemble)

    def addZCons(self, c: int):
        setZDefCons(self.mdl, self.ensemble, self.z, self.y)
        setZKlassCons(self.mdl, self.ensemble, self.z, c)

    def addZeta(self):
        self.zeta = getZeta(self.mdl)

    def addZetaCons(self, u: np.ndarray, c: int, g: int):
        setZetaCons(self.mdl, self.ensemble, self.zeta, self.y, u, c, g)

    def addObj(self):
        setZetaObj(self.mdl, self.zeta)

    def buildModel(self, u: np.ndarray, c: int, g: int):
        self.addY()
        self.addLambda()
        self.addYCons()
        self.addMu()
        self.addMuCons()
        self.addNu()
        self.addNuCons()
        self.addX()
        self.addXCons()
        self.addZ()
        self.addZCons(c)
        self.addZeta()
        self.addZetaCons(u, c, g)
        self.addObj()

    def clearY(self):
        self.y = {}

    def clearLambda(self):
        self.lam = {}

    def clearMu(self):
        self.mu = {}

    def clearNu(self):
        self.nu = {}

    def clearZ(self):
        self.z = []

    def clearZeta(self):
        self.zeta = []

    def clearModel(self):
        self.clearY()
        self.clearLambda()
        self.clearMu()
        self.clearNu()
        self.clearZ()
        self.clearZeta()
        self.mdl.clear()

    def separate(
        self,
        u: np.ndarray,
        **kwargs
    ) -> dict[tuple[int, int], np.ndarray]:
        log_output = kwargs.get('log_output', False)
        if not isinstance(log_output, bool):
            raise TypeError('log_output must be a boolean')
        
        precision = kwargs.get('precision', 8)
        if not isinstance(precision, int):
            raise TypeError('precision must be an integer')
        
        res = {}
        for c in range(self.ensemble.n_classes):
            for g in range(self.ensemble.n_classes):
                self.mdl = cpx.Model(
                    name=f'Separate_{c}_{g}',
                    log_output=log_output,
                    float_precision=precision
                )
                self.buildModel(u, c, g)
                sol = self.mdl.solve()
                if sol:
                    pass
                else:
                    pass
                self.clearModel()
        return res

class TreeEnsembleCompressor:
    dataset: pd.DataFrame
    ensemble: TreeEnsemble
    mdl: cpx.Model
    sep: TreeEnsembleSeparator
    u: list[cpv.Var]
    sol: np.ndarray
    status: str

    def __init__(
        self,
        ensemble: str | TreeEnsemble,
        dataset: str | pd.DataFrame,
    ) -> None:
        if isinstance(ensemble, str):
            ensemble = TreeEnsemble.from_file(ensemble)
        elif not isinstance(ensemble, TreeEnsemble):
            raise TypeError('ensemble must be a TreeEnsemble or a path to a file')

        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset)
        elif not isinstance(dataset, pd.DataFrame):
            raise TypeError('dataset must be a DataFrame or a path to a file')

        self.ensemble = ensemble
        self.dataset = dataset
        self.sep = TreeEnsembleSeparator(ensemble)

    def addU(self):
        self.u = getU(self.mdl, self.ensemble)

    def addUCons(self, x: np.ndarray):
        setUCons(self.mdl, self.ensemble, self.u, x)
        
    def addTrainCons(self):
        for _, x in self.dataset.iterrows():
            self.addUCons(np.array(x.values))

    def setUG(self):
        setUGCons(self.mdl, self.u)

    def setObj(self):
        setUObj(self.mdl, self.u)

    def updateSol(self):
        self.sol = np.array([v.solution_value for v in self.u])

    def buildModel(self):
        self.addU()
        self.addTrainCons()
        self.setUG()
        self.setObj()

    def compress(
        self,
        on: str = 'train',
        **kwargs
    ):
        if on not in ['train', 'full']:
            raise ValueError('on must be either "train" or "full"')

        log_output = kwargs.get('log_output', False)
        if not isinstance(log_output, bool):
            raise TypeError('log_output must be a boolean')
        
        precision = kwargs.get('precision', 8)
        if not isinstance(precision, int):
            raise TypeError('precision must be an integer')

        m_iterations = kwargs.get('m_iterations', None)
        if m_iterations is not None and not isinstance(m_iterations, int):
            raise TypeError('m_iterations must be an integer')

        self.mdl = cpx.Model(
            name="Compression",
            log_output=log_output,
            float_precision=precision
        )
        self.buildModel()
        if log_output:
            self.mdl.print_information()

        iteration = 0
        while True:
            if m_iterations is not None and iteration >= m_iterations:
                self.status = 'max_iterations'
                break        
            sol = self.mdl.solve()
            if sol:
                if log_output: self.mdl.report()
                self.updateSol()
                if self.mdl.objective_value == len(self.ensemble):
                    break
                if on == 'train':
                    self.status = 'optimal'
                    break
                elif on == 'full':
                    res = self.sep.separate(
                        self.sol,
                        log_output=log_output,
                        precision=precision
                    )
                    if len(res) == 0:
                        self.status = 'optimal'
                        break
                    else:
                        for x in res.values():
                            self.addUCons(x)
            else:
                self.status = 'infeasible'
                break
            iteration += 1

if __name__ == '__main__':
    import pathlib
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    ensemble = root / 'forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF8.txt'
    ensemble = str(ensemble)
    ensemble = TreeEnsemble.from_file(ensemble)
