import pandas as pd
import pathlib

from rfc.model import Compressor,Separator,RFC
from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    dataset = root / 'datasets/Seeds/Seeds.full.csv'
    ensemble = root / 'forests/Seeds/Seeds.RF1.txt'
    dataset = str(dataset)
    dataset = pd.read_csv(dataset)[:20]
    ensemble = str(ensemble)
    ensemble = load_tree_ensemble(ensemble, log_output=False)
    rfc = RFC(ensemble,dataset)
    rfc.solve(iterations=1000)

    

