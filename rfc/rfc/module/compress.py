import pandas as pd
import numpy as np

from enum import Enum
import docplex.mp.model as cpx
import docplex.mp.dvar as cpv

from .cplex import *

from .separate import TreeEnsembleSeparator
from ..structs.ensemble import TreeEnsemble

class TreeEnsembleCompressorStatus(Enum):
    OPTIMAL = 'optimal'
    INFEASIBLE = 'infeasible'
    UNKNOWN = 'unknown'
    MAX_ITERATIONS = 'max_iterations'
    TIME_LIMIT = 'time_limit'

class TreeEnsembleCompressor:
    dataset: pd.DataFrame
    ensemble: TreeEnsemble
    mdl: cpx.Model
    sep: TreeEnsembleSeparator
    u: list[cpv.Var]
    sol: np.ndarray
    status: TreeEnsembleCompressorStatus
    lazy: bool = False

    def __init__(
        self,
        ensemble: TreeEnsemble,
        dataset: pd.DataFrame,
        lazy: bool = False
    ) -> None:
        if not isinstance(ensemble, TreeEnsemble):
            raise TypeError('ensemble must be a TreeEnsemble')

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError('dataset must be a DataFrame')

        self.ensemble = ensemble
        self.dataset = dataset
        self.lazy = lazy
        self.sep = TreeEnsembleSeparator(ensemble)

    def addU(self):
        self.u = getU(self.mdl, self.ensemble)

    def addUCons(self, x: np.ndarray):
        setUCons(self.mdl, self.ensemble, self.u, x, self.lazy)
        
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
        sol = self.mdl.solve()
        iteration = 0
        while True:
            if m_iterations is not None and iteration >= m_iterations:
                self.status = TreeEnsembleCompressorStatus.MAX_ITERATIONS
                break        
            if sol:
                if log_output:
                    self.mdl.report()
                    print(self.mdl.solution)
                self.updateSol()
                if self.mdl.objective_value == len(self.ensemble):
                    break
                if on == 'train':
                    self.status = TreeEnsembleCompressorStatus.OPTIMAL
                    break
                elif on == 'full':
                    res = self.sep.separate(
                        self.sol,
                        log_output=log_output,
                        precision=precision
                    )
                    if len(res) == 0:
                        self.status = TreeEnsembleCompressorStatus.OPTIMAL
                        break
                    else:
                        for x in res.values():
                            self.addUCons(x)
                        sol = self.mdl.solve()
            else:
                self.status = TreeEnsembleCompressorStatus.INFEASIBLE
                break
            iteration += 1
        