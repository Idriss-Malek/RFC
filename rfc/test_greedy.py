from rfc.heuristic import GreedyCompressor
import pathlib
import pandas as pd
import time

from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    comp = lambda x,y : x<y
    with open('test_greedy.csv', 'a+') as f:
        f.write(f'Ensemble,Tuple size, Original size, Compressed size,  Original accuracy on testset , Compressed accuracy on testset, time\n')

    for word in ['FICO', 'HTRU2','Pima-Diabetes','COMPAS-ProPublica']:        
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
            greedy = GreedyCompressor(ensemble, dataset)
            for j in range(1,3):
                t1 = time.time()
                greedy.solve(iterations=100,tuples=j)
                t2 = time.time()
                acc1=0
                acc2=0
                for _,row in test_dataset.iterrows():
                    acc1 += (ensemble.klass(row,tiebreaker = comp) == row[-1]) +0. #type:ignore
                    acc2 += (ensemble.klass(row, greedy.u, tiebreaker = comp) == row[-1]) +0. #type:ignore
                acc1 /= len(test_dataset)
                acc2 /= len(test_dataset)
                
                with open('test_greedy.csv', 'a+') as f:
                    f.write(f'{word} {i}, {j}, {len(greedy.u)}, {sum(greedy.u)},{acc1} ,{acc2}, {t2 - t1} \n')
        