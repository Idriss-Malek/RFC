import numpy as np
import pandas as pd

from ..structs import TreeEnsemble

def check_on_x(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    x: np.ndarray,
    tie: bool = True
) -> list[bool]:
    w = ensemble.weights
    wu = u * w
    F = ensemble.getF(x)
    c = np.argmax(F.dot(w))
    max_wu = max(F.dot(wu))
    if tie:
        return [F.dot(wu)[c] == max(F.dot(wu)),np.count_nonzero(F.dot(wu) == max_wu) >= 2]
    else:
        return [c == np.argmax(F.dot(wu)),np.count_nonzero(F.dot(wu) == max_wu) >= 2]

def check_on_dataset(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    dataset: pd.DataFrame
) -> list[bool | float]:
    tie=0
    for _, x in dataset.iterrows():
        check = check_on_x(ensemble, u, np.array(x.values))
        tie += check[1]
        if not check[0]:
            return [False, -1.]
    return [True, tie/len(dataset)]

def rate_on_dataset(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    dataset: pd.DataFrame
):
    correct = 0
    tie=0
    for _, x in dataset.iterrows():
        check = check_on_x(ensemble, u, np.array(x.values))
        correct += check[0]
        tie += check[1]
    all = len(dataset)
    return correct / all, tie / all

def accuracy(
        ensemble: TreeEnsemble,
        dataset: pd.DataFrame,
        u: None | np.ndarray = None
) -> float:
    if u is None:
        u = np.ones(len(ensemble.trees))
    w = ensemble.weights
    wu = u * w
    acc=[]
    for _, x in dataset.iterrows():
        F = ensemble.getF(x)# type: ignore
        c = np.argmax(F.dot(wu))
        if c == x[-1]:
            acc.append(1)
        else:
            acc.append(0)

    return np.array(acc).mean()