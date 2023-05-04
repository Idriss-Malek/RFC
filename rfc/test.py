import pandas as pd
from rfc.utils import load_tree_ensemble, check_on_dataset
from rfc.module import TreeEnsembleCompressor, TreeEnsembleCompressorStatus
from docplex.mp.context import Context

from rfc.module import Model, CounterFactual
from rfc.structs import Ensemble
from rfc.structs.utils import idenumerate

def getX(E: Ensemble, v: dict):
    x = {}

    for f, _ in idenumerate(E.binary_features):
        x[f] = v[f].solution_value

    for f, F in idenumerate(E.numerical_features):
        x[f] = F.levels[0]
        k = len(F.levels)
        for j in range(1, k):
            x[f] += (F.levels[j] - F.levels[j-1]) * v[f][j].solution_value
        if v[f][k].solution_value > 0:
            x[f] = float('inf')

    return x

# Example     
if __name__ == '__main__':
    import pathlib
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    dataset = root / 'datasets/Pima-Diabetes/Pima-Diabetes.full.csv'
    ensemble = root / 'forests/Pima-Diabetes/Pima-Diabetes.RF1.txt'
    dataset = str(dataset)
    dataset = pd.read_csv(dataset)
    ensemble = str(ensemble)
    ensemble = load_tree_ensemble(ensemble, log_output=False)
    cmp = Model(log_output=False, float_precision=8)
    cmp.build(ensemble, dataset, lazy=True)
    cmp.solve()
    v = cmp.find_matching_vars('u')
    u = {t: v[t].solution_value for t, _ in idenumerate(ensemble)}
    sep = CounterFactual(log_output=True, float_precision=8)
    vars = sep.build(ensemble, u, 0, 1)
    sep.solve()
    sep.report()
    x = getX(ensemble, vars)
    print(ensemble.klass(x), ensemble.klass(x, u))