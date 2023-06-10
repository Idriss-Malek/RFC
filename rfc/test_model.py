import pandas as pd
import pathlib

from rfc.model import Compressor,Separator,RFC
from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    for i in range (1,11):
        dataset = root / f'datasets/Pima-Diabetes/Pima-Diabetes.train{i}.csv'
        test_dataset = root / f'datasets/Pima-Diabetes/Pima-Diabetes.test{i}.csv'
        ensemble = root / f'forests/Pima-Diabetes/Pima-Diabetes.RF{i}.txt'
        dataset = str(dataset)
        dataset = pd.read_csv(dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        dataset = dataset[:1]
        rfc = RFC(ensemble,dataset,test_dataset)
        rfc.solve(iterations=1000)
    for i in range (1,11):
        dataset = root / f'datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.train{i}.csv'
        test_dataset = root / f'datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.test{i}.csv'
        ensemble = root / f'forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF{i}.txt'
        dataset = str(dataset)
        dataset = pd.read_csv(dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        dataset = dataset[:1]
        rfc = RFC(ensemble,dataset,test_dataset)
        rfc.solve(iterations=1000)
    for i in range (1,11):
        dataset = root / f'datasets/Seeds/Seeds.train{i}.csv'
        test_dataset = root / f'datasets/Seeds/Seeds.test{i}.csv'
        ensemble = root / f'forests/Seeds/Seeds.RF{i}.txt'
        dataset = str(dataset)
        dataset = pd.read_csv(dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        dataset = dataset[:1]
        rfc = RFC(ensemble,dataset,test_dataset)
        rfc.solve(iterations=1000)
   