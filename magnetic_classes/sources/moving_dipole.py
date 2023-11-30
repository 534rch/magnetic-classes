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

        pos = self.path(i)
        self.dipole.x0 = pos[0]
        self.dipole.y0 = pos[1]
        self.dipole.z0 = pos[2]

        return self.dipole(x, y, z, i, dt, magnitude=magnitude)

    def getParameters(self):
        return {
            
        }
