from compresser import compress
from read_trees import nb_classes_fun, read_trees
import pandas as pd
from tree_ensemble_function import tree_ensemble_fun
import numpy as np

if __name__ == '__main__':
    current_file=str(__file__)
    data = current_file[:-15]+'resources/datasets/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.full.csv'
    rf = current_file[:-15]+'resources/forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF8.txt'
    dataset=pd.read_csv(data)
    trees=read_trees(rf)
    nb_classes=nb_classes_fun(rf)
    nb_trees=len(trees)
    u=compress(trees,dataset,nb_classes)
    new_trees=[trees[t] for t in range(len(trees)) if u[t]==1.0]
    new_nb_trees=len(new_trees)
    print(f'ORIGINAL TREE ENSEMBLE CONTAINS {nb_trees} TREES.')
    print(f'COMPRESSED TREE ENSEMBLE CONTAINS {new_nb_trees} TREES.')
    for index, row in dataset.iterrows():
        results=np.empty([nb_classes,nb_trees])
        probs=np.empty(nb_classes)
        for c in range (nb_classes):
            results[c]=tree_ensemble_fun(trees,row,c)
        probs=results.mean(axis=1)
        original_rf_class=np.argmax(probs)
        new_results=np.empty([nb_classes,new_nb_trees])
        new_probs=np.empty(nb_classes)
        for c in range (nb_classes):
            new_results[c]=tree_ensemble_fun(new_trees,row,c)
        new_probs=results.mean(axis=1)
        new__rf_class=np.argmax(new_probs)
        if original_rf_class!=new__rf_class:
            print('LOSSLESS COMPRESSION FAILED')
            exit()
    print('LOSSLESS COMPRESSION SUCCEEDED')
