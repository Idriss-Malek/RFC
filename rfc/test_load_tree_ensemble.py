from rfc.utils import load_tree_ensemble

if __name__ == '__main__':
    import pathlib
    root = pathlib.Path(__file__).parent.resolve().parent.resolve() / 'resources'
    path = root / 'forests/Breast-Cancer-Wisconsin/Breast-Cancer-Wisconsin.RF8.txt'
    path = str(path)
    ensemble = load_tree_ensemble(path, log_output=True)
    print(ensemble.features)