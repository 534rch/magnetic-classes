from abc import ABC, abstractmethod

import numpy as np

from .measurement import Measurement


class Source(ABC):
    @abstractmethod
    def __call__(self, x, y, z, i, dt, magnitude=False) -> np.ndarray or Measurement:
        pass

    @abstractmethod
    def getParameters(self):
        pass

    def tile_t(self, t, x, y, z):
        if not isinstance(t, np.ndarray):
            t = np.array([t])

        len_x = np.array(x).shape[0]
        len_t = t.shape[0]
        x = np.array(x).repeat(len_t)
        y = np.array(y).repeat(len_t)
        z = np.array(z).repeat(len_t)
        t = np.tile(t, len_x)
        return t, x, y, z, len_x, len_t


def parallel_source(source: Source, x, y, z, i, dt):
    return source(x, y, z, i, dt)
