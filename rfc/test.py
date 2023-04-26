import pandas as pd
from rfc.utils import load_tree_ensemble, check_on_dataset
from rfc.module import TreeEnsembleCompressor, TreeEnsembleCompressorStatus

# Example     
if __name__ == '__main__':
    import pathlib
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    dataset = root / 'datasets/FICO/FICO.full.csv'
    ensemble = root / 'forests/FICO/FICO.RF1.txt'
    dataset = str(dataset)
    dataset = pd.read_csv(dataset)
    ensemble = str(ensemble)
    ensemble = load_tree_ensemble(ensemble, log_output=False)
    cmp = TreeEnsembleCompressor(ensemble, dataset, lazy=True)
    cmp.compress(on='full', log_output=True, precision=8, m_iterations=5)
    if cmp.status != TreeEnsembleCompressorStatus.OPTIMAL:
        print('Solver did not find any solution.')
    elif check_on_dataset(ensemble, cmp.sol, dataset):
        print('Compression successful.')
    else:
        print('Compression failed.')