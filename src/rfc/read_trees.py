import numpy as np
import pandas as pd
from anytree import Node

def read_trees(file:str):
    trees = []
    with open(file, 'r') as f:
            next(f)
            next(f)
            nb_trees= int(f.readline().split()[1])
            nb_features = int(f.readline().split()[1])
            nb_classes = int(f.readline().split()[1])
            next(f)
            numbers = [str(i) for i in range(10)]
            for line in f:
                if line[0] in numbers:
                    
                    row=line.split()
                    node = Node(int(row[0]), feature=int(row[4]), thr=float(row[5]), klass=int(row[7]))
                    if node.name in parents: # type: ignore
                        node.parent = parents[node.name] # type: ignore
                    if int(row[2])>=0:
                        parents[int(row[2])]=node.name # type: ignore
                        parents[int(row[3])]=node.name # type: ignore
                    trees[-1].append(node)
                    
                if '[TREE' in line:
                    trees.append([])
                    parents={}


if __name__ == '__main__':
    data = '../resources/datasets/Seeds/Seeds.train1.csv'
    rf = '../resources/forests/Seeds/Seeds.RF1.txt'
    reader = Reader(rf)
    print(reader.nb_classes)
    row = pd.read_csv(data).loc[138]
    print(reader.rf_fun(row, 2))
