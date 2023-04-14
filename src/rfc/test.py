from compresser import compress
from read_trees import nb_classes_fun, read_trees
import pandas as pd
import os

if __name__ == '__main__':
    current_file=str(__file__)
    data = current_file[:-15]+'resources/datasets/Seeds/Seeds.train1.csv'
    rf = current_file[:-15]+'resources/forests/Seeds/Seeds.RF1.txt'
    dataset=pd.read_csv(data)
    trees=read_trees(rf)
    nb_classes=nb_classes_fun(rf)
    compress(trees,dataset,nb_classes)
    
