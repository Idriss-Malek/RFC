import pandas as pd
import pathlib
from time import time

from rfc.model import Compressor
from rfc.utils import load_tree_ensemble

comp = lambda x,y : x<y
with open('test_compression_train_only_1000.csv', 'a+') as f:
    f.write(f'Ensemble, Original size, Compressed size, Original accuracy on testset , Compressed accuracy on testset, time \n')

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    for word in ['FICO', 'HTRU2','Pima-Diabetes','COMPAS-ProPublica','Breast-Cancer-Wisconsin','Seeds']:
        for i in range (1,11):
            dataset = root / f'datasets/{word}/{word}.train{i}.csv'
            test_dataset = root / f'datasets/{word}/{word}.test{i}.csv'
            ensemble = root / f'forests/{word}/{word}.BIG.RF{i}.txt'
            dataset = str(dataset)
            dataset = pd.read_csv(dataset)
            test_dataset = str(test_dataset)
            test_dataset = pd.read_csv(test_dataset)
            ensemble = str(ensemble)
            ensemble = load_tree_ensemble(ensemble, log_output=False)
            t1 = time()
            compressor = Compressor(ensemble,dataset)
            compressor.build()
            compressor.solve()
            t2 = time() - t1
            acc1=0
            acc2=0
            for _,row in test_dataset.iterrows():
                acc1 += (ensemble.klass(row,tiebreaker = comp) == row[-1]) +0. #type:ignore
                acc2 += (ensemble.klass(row, compressor.u, tiebreaker = comp) == row[-1]) +0. #type:ignore
            acc1 /= len(test_dataset)
            acc2 /= len(test_dataset)
            
            with open('test_compression_train_only_1000.csv', 'a+') as f:
                f.write(f'{word} {i}, {len(compressor.u)}, {sum(compressor.u)},{acc1} ,{acc2}, {t2} \n')
