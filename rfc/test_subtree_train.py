import pandas as pd
import pathlib

from rfc.subtree import Compressor
from rfc.utils import load_tree_ensemble
from rfc.structs.utils import idenumerate

from anytree import RenderTree 

comp = lambda x,y : x<y
with open('test_subtree_compression.csv', 'a+') as f:
    f.write(f'Ensemble, Original size, Compressed size, Original nb of nodes, Compressed nb of nodes, Lossless on trainset, Original accuracy on testset , Compressed accuracy on testset \n')

def nb_trees(u,ensemble):
    s=0
    for t,tree in idenumerate(ensemble):
        if round(u[(t,tree.root.id)]) == 1:
            s+=1
    

if __name__ == '__main__':
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
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
            compressor = Compressor(ensemble,dataset)
            compressor.build()
            compressor.solve()
            compressor.new_ensemble()
            #print(RenderTree(compressor.ensemble[0].root))
            #print(RenderTree(compressor.compressed[0].root))
            #print([compressor.u[(1,v.id)] for v in ensemble[1]])
            #print([(compressor.u[(1,v.id)],v) for v in ensemble[1]])

            acc1 = 0
            acc2 = 0
            for _,row in test_dataset.iterrows():
                acc1 += (ensemble.klass(row,tiebreaker = comp) == row[-1]) +0. #type:ignore
                acc2 += (compressor.compressed.klass(row, tiebreaker = comp) == row[-1]) +0. #type:ignore
            acc1 /= len(test_dataset)
            acc2 /= len(test_dataset)
            with open('test_subtree_compression.csv', 'a+') as f:
                f.write(f' {word} {i},{len(ensemble)}, {len(compressor.compressed)}, {len(compressor.u)}, {sum(compressor.u.values())}, {compressor.check(dataset)}, {acc1} , {acc2}\n')#type:ignore
            

