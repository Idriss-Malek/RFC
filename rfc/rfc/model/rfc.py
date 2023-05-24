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
        def write_in_file(file, line):
            with open(file, 'a+') as f:
                f.write(line)
        self.compressor.build()
        initial = time.time()
        for i in range(iterations):
            self.u=self.compressor.solve()
            so_far = time.time()
            write_in_file(f'rfc_test_{initial}.csv',f"{i},{sum(self.u)},{so_far-initial},{self.compressor.check()} \n")
            '''check = self.compressor.check()
            if check <1.0:
                write_in_file(f'rfc_test_{initial}.csv',f'Lossless compression failed: {check} \n')
                print('Lossless compression failed.')
                return'''
            self.separator.update_u(self.u)
            sep=self.separator.find_all()
            if not sep:
                write_in_file(f'rfc_test_{initial}.csv','Lossless compression completed. \n')
                print('Lossless compression completed.')
                return self.u
            else:
                self.compressor.add(sep)
        self.u = self.compressor.solve()
        so_far = time.time()
        write_in_file(f'rfc_test_{initial}.csv',f"final,{sum(self.u)},{so_far-initial},{self.compressor.check()} \n")
        write_in_file(f'rfc_test_{initial}.csv',f'Number of iterations achieved with {sum(self.u)} trees. \n')
        write_in_file(f'rfc_test_{initial}.csv',f'{self.u} \n')
        print('Number of iterations achieved.')
        return self.u