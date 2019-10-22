import numpy as np
import cupy as cp
from typing import Tuple, Union

class StaticVolume:

    def __init__(self, data: Union[np.ndarray, cp.ndarray]):

        if data.ndim != 3:
            raise ValueError('Expected a 3D array')

        if isinstance(data, np.ndarray):
            self.data = cp.asarray(data)
        else:
            self.data = data


