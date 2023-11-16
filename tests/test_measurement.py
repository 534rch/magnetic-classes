from magnetic_classes import Measurement, ScalarMeasurement, VectorMeasurement
import unittest
import numpy as np

class TestMeasurement(unittest.TestCase):
    def test_init_magnitude(self):
        values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        measurement = Measurement(values, magnitude=True)
        self.assertTrue(np.array_equal(measurement(), values))
        self.assertTrue(measurement.magnitude)

    def test_init_components(self):
        values = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).T
        measurement = Measurement(values, magnitude=False)
        self.assertTrue(np.array_equal(measurement(), values))
        self.assertFalse(measurement.magnitude)

    def test_init_incorrect_shape(self):
        values = np.array([[1, 2, 3], [4, 5, 6]]).T
        with self.assertRaises(Exception):
            measurement = Measurement(values, magnitude=False)

    def test_call(self):
        values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        measurement = Measurement(values, magnitude=True)
        self.assertTrue(np.array_equal(measurement(), values))

    def test_addition(self):
        values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        measurement = Measurement(values, magnitude=True)

        values2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        measurement2 = Measurement(values2, magnitude=True)

        sum = measurement + measurement2
        
        expectedValue = np.array([[1, 2, 3, 8], [5, 6, 7, 16]]).T
        self.assertTrue(np.array_equal(sum(), expectedValue))

    def test_addition_different_shapes(self):
        values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        measurement = Measurement(values, magnitude=True)

        values2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).T
        measurement2 = Measurement(values2, magnitude=True)

        values3 = np.array([[2, 1, 3, 4], [5, 6, 7, 8]]).T
        measurement3 = Measurement(values3, magnitude=True)

        with self.assertRaises(Exception):
            sum = measurement + measurement2

        with self.assertRaises(Exception):
            sum = measurement + measurement3

    def test_subtraction(self):
        values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        measurement = Measurement(values, magnitude=True)

        values2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        measurement2 = Measurement(values2, magnitude=True)

        sum = measurement - measurement2
        
        expectedValue = np.array([[1, 2, 3, 0], [5, 6, 7, 0]]).T
        self.assertTrue(np.array_equal(sum(), expectedValue))

    def test_measurement_grid(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 20)
        z = np.linspace(0, 1, 30)
        grid = np.meshgrid(x, y, z, indexing='ij')

        B = np.random.rand(10, 20, 30)
        values = np.array([grid[0].flatten(), grid[1].flatten(), grid[2].flatten(), B.flatten()])
        measurement = Measurement(values, magnitude=True, grid=[x, y, z])

        self.assertTrue(np.array_equal(measurement.grid[0], x))
        self.assertTrue(np.array_equal(measurement.grid[1], y))
        self.assertTrue(np.array_equal(measurement.grid[2], z))


        # Assert that normal call returns the flattened values
        self.assertTrue(np.array_equal(measurement(), values))

        # Assert that the call with grid=True returns the grid
        self.assertTrue(np.array_equal(measurement(grid=True).shape[1:], grid[0].shape))

        # Assert x, y and z coordinates are the same
        self.assertTrue(np.array_equal(measurement(grid=True)[0], grid[0]))
        self.assertTrue(np.array_equal(measurement(grid=True)[1], grid[1]))
        self.assertTrue(np.array_equal(measurement(grid=True)[2], grid[2]))


class TestScalarMeasurement(unittest.TestCase):
    def test_init(self):
        values = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
        scalar_measurement = ScalarMeasurement(values, grid=True)
        self.assertTrue(np.array_equal(scalar_measurement(), values))
        self.assertTrue(scalar_measurement.magnitude)
        self.assertTrue(scalar_measurement.grid)

class TestVectorMeasurement(unittest.TestCase):
    def test_init(self):
        values = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).T
        vector_measurement = VectorMeasurement(values)
        self.assertTrue(np.array_equal(vector_measurement(), values))
        self.assertFalse(vector_measurement.magnitude)