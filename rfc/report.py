import pandas as pd
import pathlib
from time import process_time,time

from rfc.module import TreeEnsembleCompressor, TreeEnsembleCompressorStatus
from rfc.utils import check_on_dataset, rate_on_dataset, accuracy
from rfc.structs import TreeEnsemble



root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
results_dir = root / 'results'
dataset_dir = root / 'datasets'
rf_dir = root / 'forests'

def report(dataset, ensembles = None):

    if not (results_dir / dataset ).exists():
        (results_dir / dataset ).mkdir(parents=True)

    if ensembles is None:
        ensembles = [str(ensemble) for ensemble in (rf_dir / dataset).iterdir() if ensemble.is_file()]

    df = pd.DataFrame(columns=['Ensemble', 'Train Dataset', 'Test Dataset','Original Size', 'Compressed Size', 'Compression Time', 'Accuracy on Train Dataset', 'Compression is Lossless for Train dataset','Percentage of ties in train set', 'Lossless compression rate for test dataset', 'Original accuracy on Test Dataset', 'New accuracy on Test Dataset', 'Percentage of ties in test set'])
    
    for ensemble in ensembles:
        ensemble_name = ensemble[len(str(rf_dir / dataset))+1:]
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
        lossless_rate, tie_test = rate_on_dataset(ensemble, cmp.sol, test_data)

        acc_train = accuracy(ensemble, train_data)
        original_acc_test = accuracy(ensemble, test_data)
        compressed_acc_test = accuracy(ensemble, test_data, cmp.sol)

        if cmp.status != TreeEnsembleCompressorStatus.OPTIMAL:
            compression = 'NOT COMPRESSED'
        else:
            check = check_on_dataset(ensemble, cmp.sol, train_data)
            compression ,tie_train = check[0], check[1]

        row = {'Ensemble' : ensemble_name, 'Train Dataset' : train_name, 'Test Dataset' : test_name, 'Original Size' : original_size, 'Compressed Size' : compressed_size, 'Compression Time' : compression_time, 'Accuracy on Train Dataset' : acc_train, 'Compression is Lossless for Train dataset' : compression, 'Percentage of ties in train_set' : tie_train, 'Lossless compression rate for test dataset': lossless_rate, 'Original accuracy on Test Dataset' : original_acc_test,'New accuracy on Test Dataset' : compressed_acc_test, 'Percentage of ties in test set': tie_test } #type:ignore
        df = df._append(row, ignore_index = True) #type:ignore
    df.to_csv(str(results_dir / dataset/ dataset)+f'_comp{int(1000*time())}.csv') 
    
    

if __name__ == "__main__":
    report('Pima-Diabetes')
    report('FICO')
    report('Seeds')
    report('COMPAS-ProPublica')
    report('HTRU2')
    report('Breast-Cancer-Wisconsin')
    