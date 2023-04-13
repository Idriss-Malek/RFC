from reader import Reader
import cplex
import pandas as pd
import numpy as np
from anytree import Node
from typing import Any

def read_trees(
    file: str
) -> list[Node]:
    ###
    # Node:
    #   - name: str
    #   - feature: int
    #   - thr: float
    #   - left: Node
    #   - right: Node
    #   - klass: int
    nbtrees = 4
    trees = []
    for _ in range(nbtrees):
        parents = {}
        trees.append(Node(0))
        # Loop over nodes
        # for each node idx:
        #   - Check if it has a parent if yes
        #   - add it to the parent.
        #   - add its left and its right to the parents dict
        idx = 1
        left = 2
        right = 3
        feature = 4
        thr = 0.5
        klass = 7
        
        node = Node(idx, feature=feature, thr=thr, klass=klass)
        if idx in parents:
            node.parent = parents[idx]
        
        parents[left] = node
        parents[right] = node

    for t in range(nbtrees):
        root = trees[t]
    return trees

def compress(
    trees: list[Node],
    train: pd.DataFrame,
    weights: None | Any = None,
    on: str ='train'
):
    if weights == None:
        weights = np.ones(len(trees))

class Optimizer:
    def __init__(self, rf_file, dataset):
        self.rf = Reader(rf_file)
        self.dataset = pd.read_csv(dataset)

    def opt(self):
        model = cplex.Cplex()
        u = model.variables.add(names=[f"u{i}" for i in range(self.rf.nb_trees)],
                                types=[model.variables.type.binary for i in range(self.rf.nb_trees)])
        constraints = []

        if self.rf.nb_classes == 2:
            for index, row in self.dataset.iterrows():
                original_rf_class = self.rf.rf_decision(row)
                print(original_rf_class)
                constraints.append([[[f"u{i}" for i in range(self.rf.nb_trees)],
                                     [self.rf.tree_fun(row, self.rf.trees[i], original_rf_class) - self.rf.tree_fun(row,
                                                                                                                    self.rf.trees[
                                                                                                                        i],
                                                                                                                    1 - original_rf_class)
                                      for i in range(self.rf.nb_trees)]], ">=", 0.0])
        else:
            for index, row in self.dataset.iterrows():
                original_rf_class = self.rf.rf_decision(row)
                for c in range(self.rf.nb_classes):
                    constraints.append([[[f"u{i}" for i in range(self.rf.nb_trees)],
                                         [self.rf.tree_fun(row, self.rf.trees[i], original_rf_class) - self.rf.tree_fun(
                                             row,
                                             self.rf.trees[
                                                 i],
                                             c)
                                          for i in range(self.rf.nb_trees)]], ">=", 0.0])
        model.linear_constraints.add(lin_expr=constraints)
        model.objective.set_sense(model.objective.sense.minimize)
        model.objective.set_linear([(f"u{i}", 1.0) for i in range(self.rf.nb_trees)])
        model.solve()


if __name__ == '__main__':
    data = '../resources/datasets/Seeds/Seeds.train1.csv'
    rf = '../resources/forests/Seeds/Seeds.RF1.txt'
    optimizer = Optimizer(rf, data)
    optimizer.opt()
