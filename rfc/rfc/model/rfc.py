import pandas as pd

from .compressor import Compressor
from .separator import Separator

from ..structs.ensemble import Ensemble

class RFC:
    dataset : pd.DataFrame
    ensemble : Ensemble
    def __init__(
        self,
        ensemble: Ensemble,
        dataset: pd.DataFrame,
    ) -> None:
        if not isinstance(ensemble, Ensemble):
            raise TypeError('ensemble must be a TreeEnsemble')

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError('dataset must be a DataFrame')

        self.ensemble = ensemble
        self.dataset = dataset
        self.compressor = Compressor(ensemble,dataset)
        self.u = [1.0 for i in range(len(self.ensemble))]
        self.separator = Separator(ensemble,self.u)

    def solve(self,iterations = 1000):
        for i in range(iterations):
            self.u=self.compressor.solve()
            if self.compressor.check() <1.0:
                print('Lossless compression failed.')
                return
            self.separator.update_u(self.u)
            sep=self.separator.find_all()
            if not sep:
                print('Lossless compression completed.')
                return self.u
            else:
                self.compressor.add(sep)
        self.u = self.compressor.solve()
        print('Number of iterations achieved.')
        return self.u