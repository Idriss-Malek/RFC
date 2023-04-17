from compresser import compress
from read_trees import nb_classes_fun, read_trees
import pandas as pd
from tree_ensemble_function import tree_ensemble_fun
import numpy as np
import os
df=pd.DataFrame(columns=['Dataset','Random Forest', 'Original size','Compressed size','Compression time', 'Compression is lossless'])
if __name__ == '__main__':
    current_file=str(__file__)
    data_dir = [f.name for f in os.scandir(current_file[:-15]+'resources/datasets') if f.is_dir()]
    rf_dir=[f.name for f in os.scandir(current_file[:-15]+'resources/forests') if f.is_dir()]
    for subdir in data_dir:
        for i in range(1,11):
            lossless_compression=True
            data = current_file[:-15]+'resources/datasets/'+subdir+'/'+subdir+'.train'+str(i)+'.csv'
            rf = current_file[:-15]+'resources/forests/'+subdir+'/'+subdir+'.RF'+str(i)+'.txt'
            dataset=pd.read_csv(data)
            trees=read_trees(rf)
            nb_classes=nb_classes_fun(rf)
            nb_trees=len(trees)
            u,t=compress(trees,dataset,nb_classes)
            new_trees=[trees[t] for t in range(len(trees)) if u[t]==1]
            new_nb_trees=len(new_trees)
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
                    lossless_compression=False
                    break
            row={'Dataset':subdir+'.train'+str(i)+'.csv','Random Forest':subdir+'.RF'+str(i)+'.txt','Original size':nb_trees ,'Compressed size':new_nb_trees,'Compression time':f'{t/10**9}s', 'Compression is lossless':lossless_compression}
            df=df._append(row, ignore_index=True)
    df.to_csv('results.csv', index=False)

