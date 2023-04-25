import numpy as np
import pandas as pd
from ..structs import TreeEnsemble

def checkKlass(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    x: np.ndarray,
) -> bool:
    w = ensemble.weights
    wu = u * w
    F = ensemble.getF(x)
    c = np.argmax(F.dot(w))
    return F.dot(wu)[c] == max(F.dot(wu)) # type: ignore

def check(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    dataset: pd.DataFrame
) -> bool:
    for _, x in dataset.iterrows():
        if not checkKlass(ensemble, u, np.array(x.values)):
            return False
    return True

def checkRate(
    ensemble: TreeEnsemble,
    u: np.ndarray,
    dataset: pd.DataFrame
):
    res = []
    for _, x in dataset.iterrows():
        if checkKlass(ensemble, u, np.array(x.values)):
            res.append(1)
        else:
            res.append(0)
    
    return sum(res) / len(res)