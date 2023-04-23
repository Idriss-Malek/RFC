import pandas as pd
import docplex.mp.model as cpx

from model import *
from tree import *

def check_klass(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    x: np.ndarray,
) -> bool:
    w = ensemble.weights
    wu = u * w
    F = ensemble.getF(x)
    c = np.argmax(F.dot(w))
    return F.dot(wu)[c] == max(F.dot(wu)) # type: ignore

def check(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    dataset: pd.DataFrame
) -> bool:
    for _, x in dataset.iterrows():
        if not check_klass(ensemble, u, np.array(x.values)):
            return False
    return True

# Example     
if __name__ == '__main__':
    import pathlib
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    dataset = root / 'datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.full.csv'
    ensemble = root / 'forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF8.txt'
    dataset = str(dataset)
    dataset = pd.read_csv(dataset)
    ensemble = str(ensemble)
    ensemble = TreeEnsemble.from_file(ensemble)
    cmp = TreeEnsembleCompressor(ensemble, dataset)
    cmp.compress(on='train', log_output=True, precision=8)
    if cmp.status != 'optimal':
        print('Solver did not find any solution.')
    elif check(ensemble, cmp.sol, dataset):
        print('Compression successful.')
    else:
        print('Compression failed.')
