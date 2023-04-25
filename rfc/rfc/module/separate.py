from .cplex import *
from ..structs.ensemble import TreeEnsemble

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
                if c != g :
                    self.mdl = cpx.Model(
                        name=f'Separate_{c}_{g}',
                        log_output=log_output,
                        float_precision=precision
                    )
                    self.buildModel(u, c, g)
                    sol = self.mdl.solve()
                    if sol:
                        if self.mdl.objective_value < 0:
                            res[(c, g)] = np.array([self.x[i].solution_value for i in self.x.keys()])

                    else:
                        pass
                    self.clearModel()
        return res