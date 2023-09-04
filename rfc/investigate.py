import pandas as pd
import numpy as np
import pathlib
import random as rd

from rfc.model import Compressor,Separator,RFC
from rfc.utils import load_tree_ensemble
from rfc.structs.utils import idenumerate


comp = lambda x,y : x<y

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    for word in ['FICO']:
        for i in range (1,2):
            dataset = root / f'datasets/{word}/{word}.train{i}.csv'
            test_dataset = root / f'datasets/{word}/{word}.test{i}.csv'
            ensemble = root / f'forests/{word}/{word}.RF{i}.txt'
            dataset = str(dataset)
            dataset = pd.read_csv(dataset)
            test_dataset = str(test_dataset)
            test_dataset = pd.read_csv(test_dataset)
            ensemble = str(ensemble)
            ensemble = load_tree_ensemble(ensemble, log_output=False)
            for i, row in dataset.iterrows():
                if i==100:
                    break
                print(ensemble.p(row.to_numpy()))
                
                


            """compressor = Compressor(ensemble,dataset)
            compressor.build()
            compressor.solve()
            with open('investigate.csv', 'a+') as f:
                f.write(str(np.nonzero(np.array(compressor.u)))+'\n')

            for _,row in dataset.iterrows():
                s=f'{sum([(tree.klass(row) == 0)+0. for t,tree in idenumerate(ensemble)])}'#type:ignore
                for t,tree in idenumerate(ensemble):
                    s+=f',{tree.klass(row)}'#type:ignore
                with open('investigate.csv', 'a+') as f:
                    f.write(s+'\n')"""
