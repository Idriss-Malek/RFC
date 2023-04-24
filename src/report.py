import pandas as pd
import docplex.mp.model as cpx
import pathlib
from compress import check, checkRate
from time import process_time,time
from model import *
from tree import *



root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
results_dir = root / 'results'
dataset_dir = root / 'datasets'
rf_dir = root / 'forests'

def report(dataset, ensembles = None):
    if not (results_dir / dataset ).exists():
        (results_dir / dataset ).mkdir(parents=True)
    if ensembles is None:
        ensembles = [str(ensemble) for ensemble in (rf_dir / dataset).iterdir() if ensemble.is_file()]
    df = pd.DataFrame(columns=['Ensemble', 'Train Dataset', 'Test Dataset','Original Size', 'Compressed Size', 'Compression Time', 'Accuracy on Train Dataset', 'Compression is Lossless for Train dataset', 'Lossless compression rate for test dataset', 'Original accuracy on Test Dataset', 'New accuracy on Test Dataset'])
    for ensemble in ensembles:
        ensemble_name = ensemble
        with open(ensemble) as f:
            train_name=(f.readline().strip('\n'))[14:]
            f.close()
        test_name = train_name.replace("train","test")
        train_data = str(dataset_dir /dataset/ train_name )
        test_data = train_data.replace("train","test")
        ensemble = TreeEnsemble.from_file(ensemble)
        train_data=pd.read_csv(train_data)
        test_data=pd.read_csv(test_data)
        cmp = TreeEnsembleCompressor(ensemble, train_data)
        t1=process_time()
        cmp.compress(on='train', log_output=True, precision=8)
        compression_time=process_time()-t1
        original_size = len(ensemble.trees)
        compressed_size = sum(cmp.sol)
        acc_train = 'Not Implemented Yet'
        original_acc_test = 'Not Implemented Yet'
        new_acc_test = 'Not Implemented Yet'
        lossless_rate = checkRate(ensemble, cmp.sol, test_data)

        if cmp.status != 'optimal':
            compression = 'NOT COMPRESSED'
        elif check(ensemble, cmp.sol, train_data):
            compression = True
        else:
            compression = False
        row = {'Ensemble' : ensemble_name, 'Train Dataset' : train_name, 'Test Dataset' : test_name, 'Original Size' : original_size, 'Compressed Size' : compressed_size, 'Compression Time' : compression_time, 'Accuracy on Train Dataset' : acc_train, 'Compression is Lossless for Train dataset' : compression, 'Lossless compression rate for test dataset': lossless_rate, 'Original accuracy on Test Dataset' : original_acc_test,'New accuracy on Test Dataset' : new_acc_test } 
        df = df._append(row, ignore_index = True) #type:ignore
    df.to_csv(str(results_dir / dataset/ dataset)+f'_comp{time()}') 
    
    

if __name__ == "__main__":
    report('HTRU2')