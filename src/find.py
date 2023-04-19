import pandas as pd
import numpy as np
import cplex
from read_trees import read_trees

def find(original, compressed, c, weights=None):
    if weights is None:
        weights = np.ones(len(compressed))
    original_nb_trees=len(original)
    model=cplex.Cplex()
    y = {}
    for t in range(original_nb_trees):
        for v in range(len(original[t])):
            lb = 0.0
            ub = 1.0
            y[(t,v)] = model.variables.add(lb=[lb], ub=[ub], types=[model.variables.type.continuous], names=[f'y{t,v}'])
    return model


if __name__ == '__main__':
    current_file=str(__file__)
    rf = read_trees(current_file[:-11]+'resources/forests/FICO/FICO.RF1.txt')
    model=find(rf,rf,0)
    print(len(rf[0]))
    print(model.variables.get_names())


