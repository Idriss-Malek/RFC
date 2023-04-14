from anytree import Node
import numpy as np

def tree_ensemble_fun(trees:list[list[Node]], x:list, c: int):
    results=np.empty(len(trees))
