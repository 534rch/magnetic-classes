import numpy as np
from .path import Path

class Grid(Path):
    def __init__(self, dx, nx, dy, ny):
        """
        Create a grid path. Move first in x-direction, then 1 step in y-direction.
        :param dx: x-spacing of the grid.
        :param nx: number of x-points in the grid. (in total nx-1)
        :param dy: y-spacing of the grid.
        :param ny: number of y-points in the grid. (in total nx-1)
        """

        x0 = np.linspace(0, dx*(nx - 1), nx)
        y0 = np.linspace(0, dy*(ny - 1), ny)

        # Alternating mesh grid like survey lines
        # TODO: check meshgrid indexing
        x, y = np.meshgrid(x0, y0, indexing="ij")

        self.x = x.flatten()
        self.y = y.flatten()
        self.z = np.zeros_like(self.x)
        
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.grid = [x0, y0, np.array([0.0])]
