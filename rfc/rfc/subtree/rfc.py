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
        self.separator = Separator(ensemble,self.compressor.compressed,self.compressor.features)#type:ignore
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
            self.compressor.solve()
            compressed = self.compressor.new_ensemble()
            so_far = time.time()
            self.separator.update_compressed(compressed)
            sep=self.separator.find_all()
            if not sep:
                write_in_file('subtree_sep.csv',f"{i},{len(self.ensemble)},{len(compressed)},{len(self.compressor.u)},{sum(self.compressor.u.values())},{so_far-initial},{self.compressor.check()},{self.compressor.check(self.test_dataset)} \n")#type:ignore
                return compressed
            else:
                self.compressor.add(sep)
        self.compressor.solve()
        compressed = self.compressor.new_ensemble()
        so_far = time.time()
        write_in_file('subtree_sep.csv',f"{i},{len(self.ensemble)},{len(compressed)},{len(self.compressor.u)},{sum(self.compressor.u.values())},{so_far-initial},{self.compressor.check()},{self.compressor.check(self.test_dataset)} \n")#type:ignore
        return compressed
    
