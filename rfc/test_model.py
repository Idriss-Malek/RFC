import pandas as pd
import pathlib

from rfc.model import Compressor,Separator
from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    dataset = root / 'datasets/Pima-Diabetes/Pima-Diabetes.full.csv'
    ensemble = root / 'forests/Pima-Diabetes/Pima-Diabetes.RF1.txt'
    dataset = str(dataset)
    dataset = pd.read_csv(dataset)[:20]
    ensemble = str(ensemble)
    ensemble = load_tree_ensemble(ensemble, log_output=False)
    print(ensemble.features[0].levels)
    '''
    compressor = Compressor(ensemble, dataset, True)
    compressor.build()
    compressor.solve()
    compressor.check()
    print(compressor.mdl.getObjective())
    separator=Separator(ensemble, [1.0]+[0.0 for i in range(len(ensemble)-1)])
    separator.build_y()
    '''

