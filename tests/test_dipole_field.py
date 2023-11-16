import numpy as np
import unittest
from magnetic_classes import Dipole
from magnetic_classes import Field, importField
import tempfile
import time

class TestField(unittest.TestCase):
    def test_opposing_dipoles_origin(self):
        ###
        # Test two dipoles centered in origin but with opposing mx, my and mz moments
        # Expect field to be zero everywhere
        ###
        dipole1 = Dipole(0, 0, 0, 1, 2, 3)
        dipole2 = Dipole(0, 0, 0, -1, -2, -3)

        field = Field()
        field.sources = [dipole1, dipole2]

        # Test on a grid
        x = np.arange(-4, 4, 1 / 2)
        y = np.arange(-4, 4, 1 / 2)
        z = np.arange(-4, 4, 1 / 2)
        xyz = np.meshgrid(y, x, z)

        bx, by, bz = field(xyz[0], xyz[1], xyz[2]).values
        B = field(xyz[0], xyz[1], xyz[2], magnitude=True).values

        # Ensure that all values in arrays bx, by and bz are zero and B is zero
        self.assertTrue(np.all(bx == 0))
        self.assertTrue(np.all(by == 0))
        self.assertTrue(np.all(bz == 0))
        self.assertTrue(np.all(B == 0))

    def test_opposing_dipoles_mirrored(self):
        ###
        # Test two dipoles centered around origin but with opposing x positions and opposing mx, my and mz moments
        # Expect bx, by and bz to be zero everywhere on the yz-plane through the origin.
        ###

        dipole1 = Dipole(1, 0, 0, 1, 2, 3)
        dipole2 = Dipole(-1, 0, 0, -1, -2, -3)

        field = Field()
        field.sources = [dipole1, dipole2]

        # Test on a grid
        x = 0
        y = np.arange(-4, 4, 1 / 2)
        z = np.arange(-4, 4, 1 / 2)
        xyz = np.meshgrid(y, x, z)

        bx, by, bz = field(xyz[0], xyz[1], xyz[2]).values
        B = field(xyz[0], xyz[1], xyz[2], magnitude=True).values

        # Ensure that all values in arrays bx, by and bz are almost zero
        self.assertTrue(np.all(np.abs(bx) < 1e-4))
        self.assertTrue(np.all(np.abs(by) < 1e-4))
        self.assertTrue(np.all(np.abs(bz) < 1e-4))
        self.assertTrue(np.all(np.abs(B) < 1e-4))

    def test_dipole_sum(self):
        ###
        # Test if the magnitude of multiple dipoles is equal to the the magnitude of the sum of dipole moments
        ###

        dipole1 = Dipole(1, 1, 0, 1, 2, 3)
        dipole2 = Dipole(1, -1, 0, 2, 3, 4)
        dipole3 = Dipole(-1, 1, 0, 4, 3, 2)
        dipole4 = Dipole(-1, -1, 0, 3, 2, 1)
        dipole5 = Dipole(0, 0, 0, 1, 2, 3)

        field = Field()
        field.sources = [dipole1, dipole2, dipole3, dipole4, dipole5]

        # Test on a grid
        x = np.arange(-4, 4, 1 / 2)
        y = np.arange(-4, 4, 1 / 2)
        z = np.arange(-4, 4, 1 / 2)

        xyz = np.meshgrid(y, x, z)

        print(xyz[0].shape, xyz[1].shape, xyz[2].shape)
        bx, by, bz = field(xyz[0], xyz[1], xyz[2]).values
        B = field(xyz[0], xyz[1], xyz[2], magnitude=True).values

        ## Test if B and norm of bx, by and bz are equal
        err = B - np.sqrt(bx**2 + by**2 + bz**2)
        err2 = B - np.linalg.norm([bx, by, bz], axis=0)
        self.assertTrue(np.all(np.abs(err) < 1e-6))
        self.assertTrue(np.all(np.abs(err2) < 1e-7))

        ## Test if bx, by and bz are the sum of the magnetic fields of the dipoles
        bx2, by2, bz2 = np.zeros(3)
        for source in field.sources:
            tempx, tempy, tempz = source(xyz[0], xyz[1], xyz[2])
            bx2 += tempx
            by2 += tempy
            bz2 += tempz

        # Sum the magnetic field components from all dipoles
        bx3, by3, bz3 = np.sum(
            [source(xyz[0], xyz[1], xyz[2]) for source in field.sources], axis=0
        )

        self.assertTrue(np.all(np.abs(bx - bx2) < 1e-8))
        self.assertTrue(np.all(np.abs(by - by2) < 1e-8))
        self.assertTrue(np.all(np.abs(bz - bz2) < 1e-8))

        self.assertTrue(np.all(np.abs(bx - bx3) < 1e-8))
        self.assertTrue(np.all(np.abs(by - by3) < 1e-8))
        self.assertTrue(np.all(np.abs(bz - bz3) < 1e-8))

    def test_field_export_import(self):
        dipole1 = Dipole(-2, 0, -4, 0, 0, 1e9/np.pi)
        dipole2 = Dipole(2, 0, -4, 0, 0, 1e9/np.pi)
        field = Field()
        field.sources = [dipole1, dipole2]

        # Store in temporary directory
        filename = "test_field" + str(time.time()) + ".json"
        path = tempfile.gettempdir() + "/" + filename

        field.export(
            path,
            "A field consisting of two dipoles",
        )

        # Import the field
        field2 = importField(path)

        # Test if the imported field is equal to the original field
        self.assertEqual(len(field.sources), len(field2.sources))
        for i in range(len(field.sources)):
            self.assertEqual(field.sources[i].x0, field2.sources[i].x0)
            self.assertEqual(field.sources[i].y0, field2.sources[i].y0)
            self.assertEqual(field.sources[i].z0, field2.sources[i].z0)
            self.assertEqual(field.sources[i].mx, field2.sources[i].mx)
            self.assertEqual(field.sources[i].my, field2.sources[i].my)
            self.assertEqual(field.sources[i].mz, field2.sources[i].mz)