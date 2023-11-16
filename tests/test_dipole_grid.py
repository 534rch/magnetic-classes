import unittest
import tempfile
import numpy as np
import time
from magnetic_classes import DipoleGrid


class TestDipoleField(unittest.TestCase):
    def test_create_grid(self):
        grid = DipoleGrid(1, 2, 3, 1, 2, 3, 1, 1, 1)
        self.assertTrue(grid.dipoles.shape == (3, 27))

        # Assert that the grid is centered around (1,2,3) with spacings (1,2,3) and 3x3x3 grid points
        xs = 1 + 1 * np.arange(-1, 2, 1)
        ys = 2 + 2 * np.arange(-1, 2, 1)
        zs = 3 + 3 * np.arange(-1, 2, 1)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    self.assertTrue(grid.dipoles[0, i * 9 + j * 3 + k] == x)
                    self.assertTrue(grid.dipoles[1, i * 9 + j * 3 + k] == y)
                    self.assertTrue(grid.dipoles[2, i * 9 + j * 3 + k] == z)

    def test_export_grid(self):
        grid = DipoleGrid(1, 2, 3, 4, 5, 6, 7, 8, 9)

        # Store in temporary directory
        filename = "test_grid" + str(time.time()) + ".json"
        path = tempfile.gettempdir() + "/" + filename
        export_data = grid.export(path)

        # Assert that the export data is a dictionary with the correct keys
        json_data = {
            "type": "DipoleGrid",
            "description": None,
            "parameters": {
                "type": "DipoleGrid",
                "x0": grid.x0,
                "y0": grid.y0,
                "z0": grid.z0,
                "dx": grid.dx,
                "dy": grid.dy,
                "dz": grid.dz,
                "nx": grid.nx,
                "ny": grid.ny,
                "nz": grid.nz,
            },
            "seed": grid.seed
        }

        self.assertTrue(export_data == json_data)

        # Assert that the file was created
        import os
        self.assertTrue(os.path.isfile(path))

        # Assert that the file contains the correct data
        import json

        with open(path, "r") as f:
            file_data = json.load(f)
        self.assertTrue(file_data == json_data)

    # def test_grid_with_moments(self):
    #     grid = DipoleGrid(1, 2, 3, 1, 2, 3, 1, 1, 1)
    #     # Random 3x27 matrix
    #     M = np.random.rand(3, 27).T
    #     # Add column 4-6 to M
    #     grid.M = M

    #     # Place the dipoles
    #     grid.placeDipoles()

    #     # Assert that the moments are correct
    #     for i in range(27):
    #         self.assertTrue(grid.sources[i].mx == M[0, i])
    #         self.assertTrue(grid.sources[i].my == M[1, i])
    #         self.assertTrue(grid.sources[i].mz == M[2, i])