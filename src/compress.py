import pandas as pd
import docplex.mp.model as cpx

from model import *
from tree import *

def check_klass(
    ensemble: TreeEnsemble,
    u: list[int],
    x: pd.Series,
) -> bool:
    w = ensemble.weigths
    wu = np.array(u) * w
    F = ensemble.getF(x)
    c = np.argmax(F.dot(w))
    return F.dot(wu)[c] == max(F.dot(wu))

def check(
    ensemble: TreeEnsemble,
    u: list[int],
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

    with cpx.Model(
        name="Compression",
        log_output=log_output,
        float_precision=precision
    ) as mdl:
        buildModel(mdl, ensemble, dataset)
        mdl.print_information()
        u = None
        while True:
            sol = mdl.solve()
            if sol:
                if log_output: mdl.report()
                u = getSolution(mdl)
                if on == 'train':
                    return u
                elif on == 'full':
                    if sum(u) == len(ensemble.trees):
                        return u
                    else:
                        xs = separate(ensemble, u)
                        if xs == []:
                            return u
                        else:
                            for x in xs:
                                setConstraints(mdl, ensemble, x)             
            else:
                print('No solution found.')
                return None

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
    if check(ensemble, u, dataset):
        print('Compression successful.')
    else:
        print('Compression failed.')
