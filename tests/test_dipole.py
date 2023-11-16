import numpy as np
import unittest
from magnetic_classes import Dipole


class TestDipole(unittest.TestCase):
    def test_dipole_creation(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6)
        self.assertEqual(dipole.x0, 1)
        self.assertEqual(dipole.y0, 2)
        self.assertEqual(dipole.z0, 3)
        self.assertEqual(dipole.mx, 4)
        self.assertEqual(dipole.my, 5)
        self.assertEqual(dipole.mz, 6)

    def test_moment(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6)
        np.testing.assert_array_equal(dipole.moment(), np.array([4, 5, 6]))

    def test_loc(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6)
        np.testing.assert_array_equal(dipole.loc(), np.array([1, 2, 3]))

    def test_dipole_call(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6)

        [x, y, z] = [2, 3, 4]

        [mx, my, mz] = dipole(x, y, z).values

        self.assertEqual(mx, 4)

    def test_dipole_call_at_sigularity(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6)

        [x, y, z] = [1, 2, 3]

        np.testing.assert_array_equal(dipole(x, y, z).values, np.array([0, 0, 0]))

    def test_dipole_array_call_at_singularity(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6)

        [x0, y0, z0] = [1, 2, 3]
        [x1, y1, z1] = [1, 2, 4]
        [x2, y2, z2] = [2, 2, 5]

        x = np.array([x0, x1, x2])
        y = np.array([y0, y1, y2])
        z = np.array([z0, z1, z2])

        mx, my, mz = dipole(x, y, z).values

        self.assertEqual(mx[0], 0)
        self.assertEqual(my[0], 0)
        self.assertEqual(mz[0], 0)

    def test_dipole_call(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6)

        [x, y, z] = [2, 3, 5]

        [mx, my, mz] = dipole(x, y, z).values

        B = dipole(x, y, z, magnitude=True).values

        # Distance from the dipole (1,2,3) to the point (2,3,5)
        r = np.sqrt(6)

        # Weighted sum of distance and moments
        r_in_m = 1 * 4 + 1 * 5 + 2 * 6

        expected_mx = 10 ** (-7) * (3 * r_in_m / (r**5) * (1) - 4 / (r**3))
        expected_my = 10 ** (-7) * (3 * r_in_m / (r**5) * (1) - 5 / (r**3))
        expected_mz = 10 ** (-7) * (3 * r_in_m / (r**5) * (2) - 6 / (r**3))

        expected_b = np.sqrt(expected_mx**2 + expected_my**2 + expected_mz**2)

        self.assertAlmostEqual(mx, expected_mx, 20)
        self.assertAlmostEqual(my, expected_my, 20)
        self.assertAlmostEqual(mz, expected_mz, 20)
        self.assertAlmostEqual(B, expected_b, 20)

    def test_inactive_dipole(self):
        dipole = Dipole(1, 2, 3, 4, 5, 6, False)

        [x, y, z] = [2, 3, 5]

        [mx, my, mz] = dipole(x, y, z).values

        self.assertEqual(mx, 0)
        self.assertEqual(my, 0)
        self.assertEqual(mz, 0)

        dipole.enable()
        [mx, my, mz] = dipole(x, y, z).values

        self.assertNotEqual(mx, 0)
        self.assertNotEqual(my, 0)
        self.assertNotEqual(mz, 0)

        dipole.disable()
        [mx, my, mz] = dipole(x, y, z).values
        self.assertEqual(mx, 0)
        self.assertEqual(my, 0)
        self.assertEqual(mz, 0)

    def test_orthogonal_magnitude(self):
        dipole = Dipole(0, 0, 0, 0, 0, 10)
        [x, y, z] = [0, 0, 10]
        [mx, my, mz] = dipole(x, y, z).values
        b = dipole(x, y, z, magnitude=True).values

        self.assertEqual(mx, 0)
        self.assertEqual(my, 0)
        self.assertEqual(mz, b)
