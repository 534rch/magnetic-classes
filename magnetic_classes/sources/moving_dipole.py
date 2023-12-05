import numpy as np
from .source import Source
from .dipole import Dipole
from ..paths.path import Path
from .measurement import ScalarMeasurement, VectorMeasurement


class MovingDipole(Source):
    def __init__(self, dipole: Dipole, path: Path):
        self.dipole = dipole
        self.path = path

    def __call__(self, x, y, z, i, dt = 1, magnitude=False):
        """
        Calculate the magnetic field at a given point.
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :param z: z-coordinate of the point
        :param i: time index
        :param dt: time step
        :return: the magnetic field strength at the point
        """

        # Moving the dipole is the same as fixing the dipole and moving the point
        xt, yt, zt = self.path(i)

        len_x = np.array(x).shape[0]
        len_t = np.array(xt).shape[0]
        x = np.array(x).repeat(len_t)
        y = np.array(y).repeat(len_t)
        z = np.array(z).repeat(len_t)
        xt = np.tile(xt, len_x)
        yt = np.tile(yt, len_x)
        zt = np.tile(zt, len_x)

        x = x - xt
        y = y - yt
        z = z - zt


        measurement = self.dipole(x, y, z, 0, dt, magnitude=magnitude)
        measurement.t = i * dt

        return measurement

    def getParameters(self):
        return {
            
        }
