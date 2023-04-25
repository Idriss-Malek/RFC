import numpy as np
import pandas as pd

from ..structs import TreeEnsemble

def check_on_x(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    x: np.ndarray,
) -> bool:
    w = ensemble.weights
    wu = u * w
    F = ensemble.getF(x)
    c = np.argmax(F.dot(w))
    return F.dot(wu)[c] == max(F.dot(wu))

def check_on_dataset(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    dataset: pd.DataFrame
) -> bool:
    for _, x in dataset.iterrows():
        if not check_on_x(ensemble, u, np.array(x.values)):
            return False
    return True

def rate_on_dataset(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    dataset: pd.DataFrame
):
    correct = 0
    for _, x in dataset.iterrows():
        if check_on_x(ensemble, u, np.array(x.values)):
            correct += 1
    all = len(dataset)
    return correct / all