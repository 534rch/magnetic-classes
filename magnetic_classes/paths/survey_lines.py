import numpy as np
from .path import Path

class SurveyLines(Path):
    def __init__(self, dx, nx, dy, ny):
        """
        Create a survey line path. Survey lines move first in x-direction, then 1 step in y-direction, back in x-direction and so on.
        :param dx: x-spacing of the grid.
        :param nx: number of x-points in the grid. (in total 2*nx+1)
        :param dy: y-spacing of the grid.
        :param ny: number of y-points in the grid. (in total 2*ny+1)
        """

        x0 = np.linspace(0, dx*(nx - 1), nx)
        y0 = np.linspace(0, dy*(ny - 1), ny)

        # Alternating mesh grid like survey lines
        # TODO: check meshgrid indexing
        x, y = np.meshgrid(x0, y0)
        for i in range(x.shape[0]):
            if i % 2 == 0:
                x[i, :] = x[i, ::-1]
                y[i, :] = y[i, ::-1]

        self.x = x.flatten()
        self.y = y.flatten()
        self.z = np.zeros_like(self.x)
        
        # self.grid = [x0, y0, np.array([0.0])]
