import pandas as pd
import docplex.mp.model as cpx

from model import *
from tree import *

def check_klass(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    x: pd.Series,
) -> bool:
    w = ensemble.weigths
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
        if not check_klass(ensemble, u, x):
            return False
    return True

def compress(
    ensemble: str | TreeEnsemble,
    dataset: str | pd.DataFrame,
    on: str = 'train',
    **kwargs
):
    log_output = kwargs.get('log_output', False)
    if not isinstance(log_output, bool):
        raise TypeError('log_output must be a boolean')
    
    precision = kwargs.get('precision', 8)
    if not isinstance(precision, int):
        raise TypeError('precision must be an integer')

    if isinstance(ensemble, str):
        ensemble = TreeEnsemble.from_file(ensemble)
    elif not isinstance(ensemble, TreeEnsemble):
        raise TypeError('ensemble must be a TreeEnsemble or a path to a file')

    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)
    elif not isinstance(dataset, pd.DataFrame):
        raise TypeError('dataset must be a DataFrame or a path to a file')

    if on not in ['train', 'full']:
        raise ValueError('on must be either "train" or "full"')

    mdl = cpx.Model(
        name="Compression",
        log_output=log_output,
        float_precision=precision
    )
    buildModel(mdl, ensemble, dataset)
    if log_output: mdl.print_information()
    u = None
    sep = None
    while True:
        sol = mdl.solve()
        if sol:
            if log_output: mdl.report()
            u = getU(mdl)
            if on == 'train':
                return u
            elif on == 'full':
                if sep is None:
                    sep = cpx.Model(
                        name="Separation",
                        log_output=log_output,
                        float_precision=precision
                    )
                    buildBaseSepModel(sep, ensemble)
                if sum(u) == len(ensemble.trees):
                    return u
                else:
                    xs = separate(ensemble, u, baseSep=sep)
                    if xs == []:
                        return u
                    else:
                        for x in xs:
                            addCons(mdl, ensemble, x)             
        else:
            return u

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
    u = compress(ensemble, dataset, on='train', log_output=True)
    if u is None:
        print('Solver did not find any solution.')
    elif check(ensemble, u, dataset):
        print('Compression successful.')
    else:
        print('Compression failed.')
