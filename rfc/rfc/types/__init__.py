import numpy as np
import pandas as pd

Sample = np.ndarray | dict[int, float]
Dataset = pd.DataFrame

class IdentifiedObject:
    id: int