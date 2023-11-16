from abc import ABC, abstractmethod

import numpy as np

from .measurement import Measurement

class Source(ABC):
    @abstractmethod
    def __call__(self, x, y, z, magnitude=False) -> np.ndarray or Measurement:
        pass

    @abstractmethod
    def getParameters(self):
        pass
def parallel_source(source: Source, x, y, z):
    return source(x, y, z)