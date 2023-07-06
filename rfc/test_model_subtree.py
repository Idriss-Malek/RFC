import pandas as pd
import pathlib

from rfc.subtree import RFC
from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    with open('subtree_sep.csv', 'a+') as f:
        f.write('Ensemble, Iterations,Size original, Size compressed, Nb nodes original, Nb nodes compressed,compression time,lossless on train, lossless on test\n')
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
        dataset = dataset[:1]
        with open('subtree_sep.csv', 'a+') as f:
            f.write(f'Pima-Diabetes{i},')
        rfc = RFC(ensemble,dataset,test_dataset)
        rfc.solve(iterations=1000)
    for i in range (1,2):
        dataset = root / f'datasets/FICO/FICO.train{i}.csv'
        test_dataset = root / f'datasets/FICO/FICO.test{i}.csv'
        ensemble = root / f'forests/FICO/FICO.RF{i}.txt'
        dataset = str(dataset)
        dataset = pd.read_csv(dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        dataset = dataset[:1]
        with open('subtree_sep.csv', 'a+') as f:
            f.write(f'FICO{i},')
        rfc = RFC(ensemble,dataset,test_dataset)
        rfc.solve(iterations=1000)
    for i in range (1,2):
        dataset = root / f'datasets/HTRU2/HTRU2.train{i}.csv'
        test_dataset = root / f'datasets/HTRU2/HTRU2.test{i}.csv'
        ensemble = root / f'forests/HTRU2/HTRU2.RF{i}.txt'
        dataset = str(dataset)
        dataset = pd.read_csv(dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        dataset = dataset[:1]
        with open('subtree_sep.csv', 'a+') as f:
            f.write(f'HTRU2{i},')
        rfc = RFC(ensemble,dataset,test_dataset)
        rfc.solve(iterations=1000)
    for i in range (1,2):
        dataset = root / f'datasets/COMPAS-ProPublica/COMPAS-ProPublica.train{i}.csv'
        test_dataset = root / f'datasets/COMPAS-ProPublica/COMPAS-ProPublica.test{i}.csv'
        ensemble = root / f'forests/COMPAS-ProPublica/COMPAS-ProPublica.RF{i}.txt'
        dataset = str(dataset)
        dataset = pd.read_csv(dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        dataset = dataset[:1]
        with open('subtree_sep.csv', 'a+') as f:
            f.write(f'COMPAS-ProPublica{i},')
        rfc = RFC(ensemble,dataset,test_dataset)
        rfc.solve(iterations=1000)
   