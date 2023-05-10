import pandas as pd
from rfc.utils import load_tree_ensemble, check_on_dataset
from rfc.module import TreeEnsembleCompressor, TreeEnsembleCompressorStatus

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
    ensemble = root / 'forests/Pima-Diabetes/Pima-Diabetes.LITTLE.RF1.txt'
    dataset = str(dataset)
    dataset = pd.read_csv(dataset)[:20]
    ensemble = str(ensemble)
    ensemble = load_tree_ensemble(ensemble, log_output=False)
    cmp = Model(log_output=False, float_precision=8)
    cmp.build(ensemble, dataset, lazy=True)
    cmp.solve()
    print('Number of trees : ', cmp.objective_value)
    v = cmp.find_matching_vars('u')
    u = {t: v[t].solution_value for t, _ in idenumerate(ensemble)}
    all_pos=False
    iteration = 0
    while not all_pos and iteration <= 1000:
        all_pos=True
        for c in range(ensemble.n_classes):
            for g in range(ensemble.n_classes):
                if c != g:            
                    print(c,g)
                    sep = CounterFactual(log_output=True, float_precision=8)
                    vars = sep.build(ensemble, u, c, g)
                    sep.solve()
                    sep.report()
                    sep.print_information()
                    if sep.solve_details.status_code == 101: #type:ignore
                        x = getX(ensemble, vars)
                        print('Classes : ', ensemble.klass(x), ensemble.klass(x, u))
                        if sep.objective_value < 0:
                            all_pos=False
                            dataset = dataset.append_(x, ignore_index=True)#type:ignore
                    else :
                        print(sep.export_as_lp(basename='sep_fail',path='..'))
                        print('Not found optimal solution')
        cmp = Model(log_output=False, float_precision=8)
        cmp.build(ensemble, dataset, lazy=True)
        cmp.solve()
        cmp.report()
        print('Number of trees : ', cmp.objective_value) #when run, the code will give two different valeus of nmr of trees even if the dataset doesnt change.  
        v = cmp.find_matching_vars('u')
        u = {t: v[t].solution_value for t, _ in idenumerate(ensemble)}
        iteration += 1
