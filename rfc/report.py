import pandas as pd
import pathlib
from time import process_time,time

from rfc.module import TreeEnsembleCompressor, TreeEnsembleCompressorStatus
from rfc.utils import check_on_dataset, rate_on_dataset, accuracy, load_tree_ensemble



root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
results_dir = root / 'results'
dataset_dir = root / 'datasets'
rf_dir = root / 'forests'

def report(dataset, ensembles = None):

    if not (results_dir / dataset ).exists():
        (results_dir / dataset ).mkdir(parents=True)

    if ensembles is None:
        ensembles = [str(ensemble) for ensemble in (rf_dir / dataset).iterdir() if ensemble.is_file()]

    #df = pd.DataFrame(columns=['Ensemble', 'Train Dataset', 'Test Dataset','Original Size', 'Compressed Size', 'Compression Time', 'Accuracy on Train Dataset', 'Compression is Lossless for Train dataset','Percentage of ties in train set', 'Lossless compression rate for test dataset', 'Original accuracy on Test Dataset', 'New accuracy on Test Dataset', 'Percentage of ties in test set'])
    df = pd.DataFrame(columns=['Ensemble', 'Base Dataset', 'Original Size', 'Compressed Size', 'Compression Time', 'Reached Optimal ', 'Number of iterations'])
    link_csv=str(results_dir / dataset/ dataset)+f'_full_comp{int(1000*time())}.csv'
    df.to_csv(link_csv) 
    for ensemble in ensembles:
        ensemble_name = ensemble[len(str(rf_dir / dataset))+1:]
        with open(ensemble) as f:
            #train_name=(f.readline().strip('\n'))[14:]
            full_name = (f.readline().strip('\n'))[14:]
            f.close()
        #test_name = train_name.replace("train","test")
        #train_data = str(dataset_dir /dataset/ train_name )
        #test_data = train_data.replace("train","test")
        full_data = str(dataset_dir /dataset/ full_name )
        ensemble = load_tree_ensemble(ensemble)
        #train_data=pd.read_csv(train_data)
        #test_data=pd.read_csv(test_data)
        full_data = pd.read_csv(full_data)
        cmp = TreeEnsembleCompressor(ensemble, full_data)
        
        t1=process_time()
        n_iterations = cmp.compress(on='full', log_output=False, precision=8, m_iterations=1000)
        compression_time=process_time()-t1

        original_size = len(ensemble.trees)
        compressed_size = round(cmp.mdl.objective_value)
        #lossless_rate, tie_test = rate_on_dataset(ensemble, cmp.sol, test_data)

        #acc_train = accuracy(ensemble, train_data)
        #original_acc_test = accuracy(ensemble, test_data)
        #compressed_acc_test = accuracy(ensemble, test_data, cmp.sol)

        if cmp.status != TreeEnsembleCompressorStatus.OPTIMAL:
            reached = False
        else:
            reached = True
            #check = check_on_dataset(ensemble, cmp.sol, train_data)
            #compression ,tie_train = check[0], check[1]

        row = {'Ensemble' : ensemble_name, 'Base Dataset' : full_name, 'Original Size' : original_size, 'Compressed Size' : compressed_size, 'Compression Time' : compression_time, 'Reached Optimal': reached, 'Number of iterations' : n_iterations} #type:ignore
        df = df._append(row, ignore_index = True) #type:ignore
        df.to_csv(link_csv)
    
    

if __name__ == "__main__":
    report('FICO')
    