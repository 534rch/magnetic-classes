import unittest
import numpy as np
from magnetic_classes import Dipole, ScalarMeasurement, VectorMeasurement, Field, importField
import tempfile
import time

class TestField(unittest.TestCase):
    def test_init(self):
        field = Field()
        self.assertEqual(len(field.sources), 0)
        self.assertIsNotNone(field.seed)
        self.assertIsNotNone(field.M)
        self.assertIsNone(field.dipoles)

    def test_placeDipoles_no_dipoles(self):
        field = Field()
        with self.assertRaises(Exception):
            field.placeDipoles()

    def test_placeDipoles(self):
        field = Field()
        field.dipoles = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64).T
        field.M = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float64)
        field.placeDipoles()
        self.assertEqual(len(field.sources), 2)
        self.assertIsInstance(field.sources[0], Dipole)
        self.assertIsInstance(field.sources[1], Dipole)

    def test_call(self):
        field = Field()
        field.dipoles = np.array([[0, 0, 0, 1], [1, 1, 1, -1]], dtype=np.float64).T
        field.M = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float64)
        field.placeDipoles()
        measurement = field(0, 0, 0)
        self.assertIsInstance(measurement, VectorMeasurement)

    def test_dipole_moments(self):
        M = np.random.rand(3).reshape((1, 3))


        field = Field()
        # Place dipoles at different depths (z)
        zs = np.linspace(0, 10, 10)
        field.dipoles = np.array([[0,0,z] for z in zs], dtype=np.float64).T
        field.M = M
        field.placeDipoles()
        
        # Check that the dipole moments are scaled correctly
        for i in range(len(zs)):
            dipole = field.sources[i]

            self.assertAlmostEqual(dipole.mx, M[0,0]*np.abs(zs[i])**3*1e9, places=3)
            self.assertAlmostEqual(dipole.my, M[0,1]*np.abs(zs[i])**3*1e9, places=3)
            self.assertAlmostEqual(dipole.mz, M[0,2]*np.abs(zs[i])**3*1e9, places=3)

    def test_dipole_moments_seed(self):
        seed = 1234
        field = Field(seed=seed)

        M = np.array([[0.19151945, 0.62210877, 0.43772774]])
        self.assertEqual(field.seed, seed)
        self.assertTrue(np.allclose(field.M, M))
        

    def test_call_scalar(self):
        field = Field()
        field.dipoles = np.array([[0, 0, 0, 1], [1, 1, 1, -1]], dtype=np.float64).T
        field.M = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float64)
        field.placeDipoles()
        measurement = field(0, 0, 0, magnitude=True)
        self.assertIsInstance(measurement, ScalarMeasurement)

    def test_call_vector(self):
        field = Field()
        field.dipoles = np.array([[0, 0, 0, 1], [1, 1, 1, -1]], dtype=np.float64).T
        field.M = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float64)
        field.placeDipoles()
        measurement = field(0, 0, 0, magnitude=False)
        self.assertIsInstance(measurement, VectorMeasurement)

    def test_export(self):
        # Store in temporary directory
        filename = "test_field" + str(time.time()) + ".json"
        path = tempfile.gettempdir() + "/" + filename
        
        field = Field()
        field.dipoles = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64).T
        field.M = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float64)
        field.placeDipoles()
        data = field.export(path)
        self.assertEqual(data["type"], "Field")
        self.assertEqual(len(data["parameters"]), 2)
        self.assertIsNotNone(data["seed"])

        # Also check that the file exists
        with open(path, "r") as f:
            assert f.read() != ""

        return path

    def test_export_and_importField(self):
        filename = "test_field" + str(time.time()) + ".json"
        path = tempfile.gettempdir() + "/" + filename

        field = Field()
        field.dipoles = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64).T
        field.M = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float64)
        field.placeDipoles()
        field.export(path)


        field = importField(path)
        self.assertIsInstance(field, Field)
        self.assertEqual(len(field.sources), 2)
        self.assertIsInstance(field.sources[0], Dipole)
        self.assertIsInstance(field.sources[1], Dipole)
