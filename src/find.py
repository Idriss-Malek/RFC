import pandas as pd
import numpy as np
import cplex
from read_trees import read_trees, nb_classes_fun
import time

def solve_milp(original, u, klass,c, nb_classes, weights=None):
    if weights is None:
        weights = np.ones(len(original))
    original_nb_trees=len(original)
    model=cplex.Cplex()
    y={}
    lbda= {}
    p = {}
    for t in range(original_nb_trees):
        for v in original[t]:
            lb = 0.0
            ub = 1.0
            y[(t,v)]= model.variables.add(lb=[lb], ub=[ub], types=[model.variables.type.continuous], names=[f'y{t,v}'])
            for c in range(nb_classes):
                 p[(t,v,c)]=(v.klass==c)+0.
        for d in range(original[t][0].height):
            lbda[(t,d)] = model.variables.add(types=[model.variables.type.binary], names=[f'lbda{t,d}'])
    constraints=[]
    senses=[]
    rhs=[]

    for c in range(nb_classes): #constraint klass is the original chosen class
        if c!=klass:
            constraints.append([[y[(t,v)][0] for t in range(original_nb_trees) for v in original[t] if v.is_leaf],
                            [weights[t]*(p[(t,v,klass)] -p[(t,v,c)]) for t in range(original_nb_trees) for v in original[t] if v.is_leaf]])
            senses.append('G')
            rhs.append(0.)
    
    for t in range(len(original)): #constraint 6
        constraints.append([[y[(t,original[t][0])][0]],[1.]])
        senses.append('E')
        rhs.append(1.)
    for t in range(len(original)): #constraint 7
        for v in original[t]:
            if not v.is_leaf:
                constraints.append([[y[(t,v)][0],y[(t,v.children[0])][0],y[(t,v.children[1])][0]],[1.,-1.,-1.]])
                senses.append('E')
                rhs.append(0.)
    for t in range(len(original)): #constraint 8
        for d in range(original[t][0].height):
            constraints.append([[lbda[(t,d)][0]]+[y[(t,v.children[0])][0] for v in original[t] if ((not v.is_leaf) and v.depth==d)],[1.]+[-1. for v in original[t] if ((not v.is_leaf) and v.depth==d)]])
            senses.append('G')
            rhs.append(0.)
    model.linear_constraints.add(lin_expr=constraints,senses = senses,rhs = rhs)
    model.objective.set_sense(model.objective.sense.minimize)

    objective=[]
    for t in range(len(original)):
        for v in original[t][0].leaves:
            objective.append((y[(t,v)][0],u[t]*weights[t]*(p[(t,v,klass)]-p[(t,v,c)])))
    
    model.objective.set_linear(objective)
    t1=time.perf_counter_ns()
    model.solve()
    t2=time.perf_counter_ns()-t1
    if model.solution.get_objective_value()>=0:
        return True
    else:
        return model.solution.get_values(),t2


if __name__ == '__main__':
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    current_file=str(__file__)
    rf = read_trees(current_file[:-11]+'resources/forests/Pima-Diabetes/Pima-Diabetes.RF1.txt')
    result=solve_milp(rf,[1.0,1.0,0.,0.,0.,0.,0.,0.,0.,0.],0,1,2)
    print(result)


