import numpy as np
from .field import Field


class DipoleGrid(Field):
    def __init__(self, x0, y0, z0, dx, dy, dz, nx, ny, nz, **kwargs):
        """
        :param x0: x-coordinate of the center of the dipole grid
        :param y0: y-coordinate of the center of the dipole grid
        :param z0: z-coordinate of the center of the dipole grid
        :param dx: step size in x-direction
        :param dy: step size in y-direction
        :param dz: step size in z-direction
        :param nx: number of grid points in x-direction (in total 2*nx+1)
        :param ny: number of grid points in y-direction (in total 2*ny+1)
        :param nz: number of grid points in z-direction (in total 2*nz+1)
        """
        super().__init__()

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.dipoles = np.mgrid[
            -nx * dx + x0 : (nx + 1) * dx + x0 : dx,
            -ny * dy + y0 : (ny + 1) * dy + y0 : dy,
            -nz * dz + z0 : (nz + 1) * dz + z0 : dz,
        ]
        self.dipoles = self.dipoles.reshape(3, -1)


    def getParameters(self):
        """
        :return: a dictionary with the parameters of the grid
        """
        return {
            "type": self.__class__.__name__,
            "x0": self.x0,
            "y0": self.y0,
            "z0": self.z0,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
        }
