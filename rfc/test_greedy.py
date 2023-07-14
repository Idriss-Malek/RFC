from rfc.heuristic import GreedyCompressor
import pathlib
import pandas as pd

from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    for i in range (1,2):
        dataset = root / f'datasets/Pima-Diabetes/Pima-Diabetes.train{i}.csv'
        test_dataset = root / f'datasets/Pima-Diabetes/Pima-Diabetes.test{i}.csv'
        ensemble = root / f'forests/Pima-Diabetes/Pima-Diabetes.RF{i}.txt'
        dataset = str(dataset)
        dataset = pd.read_csv(dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        greedy = GreedyCompressor(ensemble, dataset)
        greedy.solve()
        