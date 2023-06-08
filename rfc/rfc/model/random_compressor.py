import random as rd
import pandas as pd

from ..structs.ensemble import Ensemble

from .compressor import Compressor


class RandomCompressor:
    ensemble : Ensemble
    def __init__(
        self,
        ensemble: Ensemble,
    ) -> None:
        if not isinstance(ensemble, Ensemble):
            raise TypeError('ensemble must be a TreeEnsemble')
        self.ensemble = ensemble
        self.df = pd.DataFrame()
        self.compressor = Compressor(ensemble,self.df)
    def pick_one(self):
        x= {}
        for feature in self.ensemble.features:
            if feature.isnumerical():
                levels = [feature.levels[-1]-1]+feature.levels+[feature.levels[-1] + 1]
                choice = rd.randint(0,len(levels)-2)
                x[feature.id] = (levels[choice]+levels[choice+1])/2
            if feature.isbinary():
                x[feature.id] = rd.randint(0,1)
        return x
    def pick_dataset(self, nb):
        rows = []
        for i in range(nb):
            rows.append(self.pick_one())
        df = pd.DataFrame.from_records(rows)
        self.df = df    
    
    def add_df(self, nb):
        rows = []
        for i in range(nb):
            rows.append(self.pick_one())
        self.df = self.df._append(rows, ignore_index=True)#type: ignore
    
    def solve(self):
        self.compressor.update_dataset(self.df)
        self.compressor.build()
        self.compressor.solve()

    



