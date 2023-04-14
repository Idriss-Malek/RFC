import numpy as np
import pandas as pd
from anytree import Node


'''
This file is meant to put the random forest that are available in the resources folder to the right format.
'''

def read_trees(file:str):
    trees = []
    with open(file, 'r') as f:
            for _ in range (6):
                next(f)
            numbers = [str(i) for i in range(10)]
            for line in f:
                if line[0] in numbers:
                    row=line.split()
                    node = Node(int(row[0]), feature=int(row[4]), thr=float(row[5]), klass=int(row[7]))
                    if node.name in parents: # type: ignore
                        node.parent = parents[node.name] # type: ignore
                    if int(row[2])>=0:
                        parents[int(row[2])]=node # type: ignore
                        parents[int(row[3])]=node # type: ignore
                    trees[-1].append(node)
                    
                if '[TREE' in line:
                    trees.append([])
                    parents={}
    return trees

def nb_classes_fun(file:str):
    with open(file, 'r') as f:
        for _ in range (4):
            next(f)
        nb_classes = int(f.readline().split()[1])
    return nb_classes

if __name__ == '__main__':
    rf = './resources/forests/Seeds/Seeds.RF1.txt'
    print(read_trees(rf)[0])

