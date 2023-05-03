import pandas as pd
from rfc.utils import load_tree_ensemble, check_on_dataset
from rfc.module import TreeEnsembleCompressor, TreeEnsembleCompressorStatus

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
    cmp = TreeEnsembleCompressor(ensemble, dataset, lazy=True)
    cmp.compress(on='train', log_output=True, precision=8, m_iterations=10)
    res=cmp.sep.separate(cmp.sol)
    x=res[(0,1)]
    print('NEW CLASS : ',ensemble.klass(x,list(cmp.sol)))
    print('OLD CLASS : ', ensemble.klass(x))   
    