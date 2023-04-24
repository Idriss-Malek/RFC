import pathlib
from compress import checkKlass, check

root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
results_dir = root / 'results'
dataset_dir = root / 'datasets'
rf_dir = root / 'forests'

def report(dataset):
    if not (results_dir / dataset ).exists():
        (results_dir / dataset ).mkdir(parents=True)
    data = str(dataset_dir / dataset / dataset)+ ''.full.csv'
    rf = root / 'forests/FICO/FICO.RF1.txt'
    


if __name__ == "__main__":
    report()