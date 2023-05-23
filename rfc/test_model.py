import pandas as pd
import pathlib

from rfc.model import Compressor,Separator
from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    dataset = root / 'datasets/FICO/FICO.full.csv'
    ensemble = root / 'forests/FICO/FICO.RF1.txt'
    dataset = str(dataset)
    dataset = pd.read_csv(dataset)[:20]
    ensemble = str(ensemble)
    ensemble = load_tree_ensemble(ensemble, log_output=False)
    separator=Separator(ensemble, [1.0]+[0.0 for i in range(len(ensemble)-1)])
    rows=separator.find_all()
    print('SEPARATOR IS : ',rows)

    

