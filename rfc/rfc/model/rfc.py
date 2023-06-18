import pandas as pd
import time 

from .compressor import Compressor
from .separator import Separator

from ..structs.ensemble import Ensemble

class RFC:
    ensemble : Ensemble
    dataset : pd.DataFrame
    def __init__(
        self,
        ensemble: Ensemble,
        dataset: pd.DataFrame,
        test_dataset : pd.DataFrame | None = None
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
        self.test_dataset = test_dataset


    def check(self, dataset = None):
        self.compressor.check(dataset)

    def solve(self,iterations = 1000):
        def write_in_file(file, line):
            with open(file, 'a+') as f:
                f.write(line)
        self.compressor.build()
        initial = time.time()
        for i in range(iterations):
            self.u=self.compressor.solve()
            so_far = time.time()
            self.separator.update_u(self.u)
            sep=self.separator.find_all()
            if not sep:
                write_in_file(f'rfc_test3.csv',f"{len(self.dataset)},{i},{sum(self.u)},{so_far-initial},{self.compressor.check()},{self.compressor.check(self.test_dataset)} \n")
                #if not self.test_dataset is None:
                    #write_in_file(f'rfc_test_{initial}.csv',f'Lossless compression on test_dataset : {self.check(self.test_dataset)}')
                return self.u
            else:
                self.compressor.add(sep)
        self.u = self.compressor.solve()
        so_far = time.time()
        write_in_file(f'compression_fullspace.csv',f"{len(self.dataset)},{iterations},{sum(self.u)},{so_far-initial},{self.compressor.check()},{self.compressor.check(self.test_dataset)} \n")
        return self.u
    
