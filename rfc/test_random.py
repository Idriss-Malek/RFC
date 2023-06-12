import pandas as pd
import pathlib
import time

from rfc.model import RandomCompressor
from rfc.utils import load_tree_ensemble

nb = 1000

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    for i in range (1,11):
        train_dataset = root / f'datasets/FICO/FICO.train{i}.csv'
        test_dataset = root / f'datasets/FICO/FICO.test{i}.csv'
        ensemble = root / f'forests/FICO/FICO.RF{i}.txt'
        train_dataset = str(train_dataset)
        train_dataset = pd.read_csv(train_dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        t1=time.time()
        rc = RandomCompressor(ensemble)
        rc.pick_dataset(nb)
        rc.solve()
        t2=time.time()
        with open('random_compression_report1.csv', 'a+') as f:
                f.write(f'\nFICO {i},{nb},{sum(rc.compressor.u)},{t2-t1},{rc.compressor.check(train_dataset,True)},{rc.compressor.check(test_dataset,True)}')
    
    for i in range (1,11):
        train_dataset = root / f'datasets/COMPAS-ProPublica/COMPAS-ProPublica.train{i}.csv'
        test_dataset = root / f'datasets/COMPAS-ProPublica/COMPAS-ProPublica.test{i}.csv'
        ensemble = root / f'forests/COMPAS-ProPublica/COMPAS-ProPublica.RF{i}.txt'
        train_dataset = str(train_dataset)
        train_dataset = pd.read_csv(train_dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        t1=time.time()
        rc = RandomCompressor(ensemble)
        rc.pick_dataset(nb)
        rc.solve()
        t2=time.time()
        with open('random_compression_report1.csv', 'a+') as f:
                f.write(f'\nCOMPAS-ProPublica {i},{nb},{sum(rc.compressor.u)},{t2-t1},{rc.compressor.check(train_dataset,True)},{rc.compressor.check(test_dataset,True)}')

    for i in range (1,11):
        train_dataset = root / f'datasets/HTRU2/HTRU2.train{i}.csv'
        test_dataset = root / f'datasets/HTRU2/HTRU2.test{i}.csv'
        ensemble = root / f'forests/HTRU2/HTRU2.RF{i}.txt'
        train_dataset = str(train_dataset)
        train_dataset = pd.read_csv(train_dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        t1=time.time()
        rc = RandomCompressor(ensemble)
        rc.pick_dataset(nb)
        rc.solve()
        t2=time.time()
        with open('random_compression_report1.csv', 'a+') as f:
                f.write(f'\nHTRU2 {i},{nb},{sum(rc.compressor.u)},{t2-t1},{rc.compressor.check(train_dataset,True)},{rc.compressor.check(test_dataset,True)}')
    for i in range (1,11):
        train_dataset = root / f'datasets/Pima-Diabetes/Pima-Diabetes.train{i}.csv'
        test_dataset = root / f'datasets/Pima-Diabetes/Pima-Diabetes.test{i}.csv'
        ensemble = root / f'forests/Pima-Diabetes/Pima-Diabetes.RF{i}.txt'
        train_dataset = str(train_dataset)
        train_dataset = pd.read_csv(train_dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        t1=time.time()
        rc = RandomCompressor(ensemble)
        rc.pick_dataset(nb)
        rc.solve()
        t2=time.time()
        with open('random_compression_report1.csv', 'a+') as f:
                f.write(f'\nPima-Diabetes {i},{nb},{sum(rc.compressor.u)},{t2-t1},{rc.compressor.check(train_dataset,True)},{rc.compressor.check(test_dataset,True)}')
    for i in range (1,11):
        train_dataset = root / f'datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.train{i}.csv'
        test_dataset = root / f'datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.test{i}.csv'
        ensemble = root / f'forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF{i}.txt'
        train_dataset = str(train_dataset)
        train_dataset = pd.read_csv(train_dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        t1=time.time()
        rc = RandomCompressor(ensemble)
        rc.pick_dataset(nb)
        rc.solve()
        t2=time.time()
        with open('random_compression_report1.csv', 'a+') as f:
                f.write(f'\nFICO {i},{nb},{sum(rc.compressor.u)},{t2-t1},{rc.compressor.check(train_dataset,True)},{rc.compressor.check(test_dataset,True)}')
    for i in range (1,11):
        train_dataset = root / f'datasets/Seeds/Seeds.train{i}.csv'
        test_dataset = root / f'datasets/Seeds/Seeds.test{i}.csv'
        ensemble = root / f'forests/Seeds/Seeds.RF{i}.txt'
        train_dataset = str(train_dataset)
        train_dataset = pd.read_csv(train_dataset)
        test_dataset = str(test_dataset)
        test_dataset = pd.read_csv(test_dataset)
        ensemble = str(ensemble)
        ensemble = load_tree_ensemble(ensemble, log_output=False)
        t1=time.time()
        rc = RandomCompressor(ensemble)
        rc.pick_dataset(nb)
        rc.solve()
        t2=time.time()
        with open('random_compression_report1.csv', 'a+') as f:
                f.write(f'\nFICO {i},{nb},{sum(rc.compressor.u)},{t2-t1},{rc.compressor.check(train_dataset,True)},{rc.compressor.check(test_dataset,True)}')