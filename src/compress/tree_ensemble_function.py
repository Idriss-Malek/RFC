from anytree import Node
import numpy as np

def tree_ensemble_fun(trees, x, c):
    '''
    The tree ensemble is a list of trees.
    Each tree is a list of nodes.
    The first node of each tree should be it's root.
    '''
    results=np.empty(len(trees))
    for i in range(len(trees)):
        node=trees[i][0]
        while node.klass==-1:
            if x[node.feature]<=node.thr:
                node=node.children[0]
            else:
                node=node.children[1]
        results[i]=(c==node.klass)+0.
    return results
