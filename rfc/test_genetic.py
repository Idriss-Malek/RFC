from rfc.heuristic import Genetic
import pathlib
import pandas as pd
import time

from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    comp = lambda x,y : x<y
    with open('test_genetic.csv', 'a+') as f:
        f.write(f'Ensemble, Original size, Compressed size,  Original accuracy on testset , Compressed accuracy on testset, time\n')

    #for word in ['FICO', 'HTRU2','Pima-Diabetes','COMPAS-ProPublica']:  
    for word in [ 'HTRU2']:
        for i in range (1,11):
            dataset = root / f'datasets/{word}/{word}.train{i}.csv'
            test_dataset = root / f'datasets/{word}/{word}.test{i}.csv'
            ensemble = root / f'forests/{word}/{word}.RF{i}.txt'
            dataset = str(dataset)
            dataset = pd.read_csv(dataset)
            test_dataset = str(test_dataset)
            test_dataset = pd.read_csv(test_dataset)
            ensemble = str(ensemble)
            ensemble = load_tree_ensemble(ensemble, log_output=False)
            genetic = Genetic(ensemble, dataset)
            print(word , i)
            t1 = time.time()
            genetic.genetic()
            t2 = time.time()
            acc1=0
            acc2=0
            for _,row in test_dataset.iterrows():
                acc1 += (ensemble.klass(row,tiebreaker = comp) == row[-1]) +0. #type:ignore
                acc2 += (ensemble.klass(row, genetic.u, tiebreaker = comp) == row[-1]) +0. #type:ignore
            acc1 /= len(test_dataset)
            acc2 /= len(test_dataset)
            with open('test_greedy.csv', 'a+') as f:
                f.write(f'{word} {i}, {genetic.n_trees}, {genetic.n_trees - genetic.best + 1},{acc1} ,{acc2}, {t2 - t1} \n')