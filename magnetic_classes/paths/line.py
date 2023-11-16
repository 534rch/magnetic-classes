import numpy as np
from .path import Path

class Line(Path):
    def __init__(self, dx, dy, n):
        """
        Create a line path. Move in direction (dx, dy) n times.
        :param dx: x-spacing of the line.
        :param dy: y-spacing of the line.
        :param n: number of points in the line.
        """

        x0 = np.linspace(0, dx*(n - 1), n)
        y0 = np.linspace(0, dy*(n - 1), n)

        self.x = x0
        self.y = y0
        self.z = np.zeros_like(self.x)

        self.dx = dx
        self.dy = dy
        self.n = n