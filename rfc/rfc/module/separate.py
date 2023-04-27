from .cplex import *
from ..structs.ensemble import TreeEnsemble

class TreeEnsembleSeparator:
    ensemble: TreeEnsemble
    mdl: cpx.Model
    y: dict[tuple[int, int], cpv.Var]
    lam: dict[tuple[int, int], cpv.Var]
    z: list[cpv.Var]
    zeta: list[cpv.Var]
    mu: dict[tuple[int, int], cpv.Var]
    nu: dict[tuple[int, int], cpv.Var]
    xi: dict[int, cpv.Var]
    x: np.ndarray

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

    def addXi(self):
        self.xi = getXi(self.mdl, self.ensemble)

    def addXiCons(self):
        setXiBinaryCons(self.mdl, self.ensemble, self.xi, self.y)

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

    def updateX(self):
        n_features = len(self.ensemble.features)
        self.x = np.zeros(n_features)
        for feature in self.ensemble.features:
            f = feature.id
            match feature.ftype:
                case FeatureType.BINARY:
                    self.x[f] = self.xi[f].solution_value    
                case FeatureType.NUMERICAL:
                    levels = feature.levels
                    k = len(levels) + 1
                    vs = [0.5 * (1 + np.tanh(v)) for v in levels]
                    vs = [0.0] + vs + [1.0]
                    vs = np.array(vs)
                    mu = np.array([self.mu[(f, j)].solution_value for j in range(k)])
                    self.x[f] = np.arctanh(2 * (np.diff(vs) @ mu) - 1) # TODO: check this later.
                    pass
                case FeatureType.CATEGORICAL:
                    pass

    def buildModel(self, u: np.ndarray, c: int, g: int):
        self.addY()
        self.addLambda()
        self.addYCons()
        self.addMu()
        self.addMuCons()
        self.addNu()
        self.addNuCons()
        self.addXi()
        self.addXiCons()
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
        Ys = []
        Xs = []
        for c in range(self.ensemble.n_classes):
            for g in range(self.ensemble.n_classes):
                if c != g :
                    self.mdl = cpx.Model(
                        name=f'Separate_{c}_{g}',
                        log_output=log_output,
                        float_precision=precision
                    )
                    self.buildModel(u, c, g)
                    if log_output: self.mdl.print_information()
                    sol = self.mdl.solve()
                    if sol:
                        if log_output:
                            self.mdl.report()
                        if self.mdl.objective_value < 0:
                            self.updateX()
                            cc = np.argmax(self.ensemble.getF(self.x).dot(self.ensemble.weights))
                            gg = np.argmax(self.ensemble.getF(self.x).dot(self.ensemble.weights * u))
                            print(self.ensemble.getF(self.x).dot(self.ensemble.weights), self.ensemble.getF(self.x).dot(self.ensemble.weights * u))
                            print(self.x)
                            for z in self.z:
                                print(f'{z.name} : {z.solution_value}')
                            for zeta in self.zeta:
                                print(f'{zeta.name} : {zeta.solution_value}')
                            Ys.append([y.solution_value for key,y in self.y.items()])
                            Xs.append(self.x)
                            res[(c, g)] = self.x.copy()
                    else:
                        pass
                    self.clearModel()
        print('y : ',Ys[0] == Ys[1])
        print('x : ',Xs[0] == Xs[1])
        return res