from compresser import compress
from read_trees import nb_classes_fun, read_trees
import pandas as pd


if __name__ == '__main__':
    data = '.../resources/datasets/Seeds/Seeds.train1.csv'
    rf = '.../resources/forests/Seeds/Seeds.RF1.txt'
    dataset=pd.read_csv(data)
    trees=read_trees(rf)
    nb_classes=nb_classes_fun(rf)
    compress(trees,dataset,nb_classes)
    
