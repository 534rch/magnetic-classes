from matplotlib import pyplot as plt
from matplotlib import tri
import numpy as np
import copy
class Measurement:
    def __init__(self, measurements, magnitude=False, **kwargs):
        """
        :param measurements: a list of measurements (x, y, z, Bx, By, Bz) or (x, y, z, B) (see magnitude)
        :param magnitude: if True, the magnitude of the magnetic field is used instead of the components
        :param **kwargs: additional arguments
        """
        
        # Check if measurements has the correct shape (n,6) or (n,4)
        # print("measurements.shape = {}".format(measurements.shape))

        if magnitude:
            if measurements.shape[0] != 4:
                raise Exception("The shape of the measurements is incorrect. Expected (4,n) but got {}".format(measurements.shape))
            self.values = measurements[3]
        else:
            if measurements.shape[0] != 6:
                raise Exception("The shape of the measurements is incorrect. Expected (6,n) but got {}".format(measurements.shape))
            self.values = measurements[3:6]

        # Assert that the measurements are flattened
        # print(measurements)
        if len(measurements.shape) == 1:
            measurements = measurements.reshape((1, measurements.shape[0]))
        if len(measurements.shape) != 2:
            raise Exception("The measurements must be flattened.")
            
        self.measurements = measurements
        self.magnitude = magnitude

        self.grid = None
        if kwargs.get("grid") is not None:
            self.grid : np.ndarray = kwargs.get("grid")

    def __call__(self, grid = False, check_grid = True):
        """
        Return the measurements.
        """
        if grid:
            # Check if the x and y coordinates are the same as the flattened grid
            grid = np.meshgrid(*self.grid, indexing="ij")
            shape = (len(self.grid[0]), len(self.grid[1]), len(self.grid[2]))
            if check_grid:
                if not np.all(self.measurements[0] == grid[0].flatten()):
                    raise Exception("The x coordinates are not the same as the flattened grid.")
                if not np.all(self.measurements[1] == grid[1].flatten()):
                    raise Exception("The y coordinates are not the same as the flattened grid.")
                if self.grid[2] is not None and not np.all(self.measurements[2] == grid[2].flatten()):
                    raise Exception("The z coordinates are not the same as the flattened grid.")
            
            # Reshape the measurements from (4,n) or (6,n) to (4,n_x,n_y,n_z) or (6,n_x,n_y,n_z)
            return self.measurements.reshape((self.measurements.shape[0], *shape))

        return self.measurements
    
    def __add__(self, other):
        """
            Add the magnetic field values.
        """

        # Check if both fields have the same length of measurements
        if self.measurements.shape != other.measurements.shape:
            raise Exception("The measurements have different lengths. Cannot add them.")
        
        # Check if all coordinates are the same
        if not np.all(self.measurements[0:3] == other.measurements[0:3]):
            raise Exception("The measurements have different coordinates. Cannot add them.")
        
        measurement = copy.deepcopy(self)

        measurement.measurements[3:] = self.measurements[3:] + other.measurements[3:]

        return measurement
    
    def __neg__(self):
        """
            Negate the magnetic field values.
        """

        measurement = copy.deepcopy(self)
        measurement.measurements[3:] = -measurement.measurements[3:]
        return measurement
    
    def __sub__(self, other):
        """
            Subtract the magnetic field values.
        """
    
        return self + (-other)
        
    
    def plot(self, fig = None, ax = None, pos = 111, **kwargs):
        """
        Plot the measurements in the x-y plane.
        """

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(pos, title=kwargs.get("title"))

        if self.magnitude:
            triang = tri.Triangulation(self.measurements[0], self.measurements[1])

            trip = ax.tripcolor(triang, self.measurements[3], cmap='viridis')

        else:
            triang = tri.Triangulation(self.measurements[0], self.measurements[1])
            if kwargs.get("component") is None:
                magnitude = np.linalg.norm(self.measurements[3:], axis=0)

                trip = ax.tripcolor(triang, magnitude, cmap='viridis', vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
            
            else:
                trip = ax.tripcolor(triang, self.measurements[3 + kwargs.get("component")], cmap='viridis', vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))

        if kwargs.get("hide_colorbar") is None:
            cbar = plt.colorbar(trip)
            cbar.set_label("B [nT]")

        if kwargs.get("hide_x"):
            ax.set_xticks([])
        if kwargs.get("hide_y"):
            ax.set_yticks([])

    def scatter(self, fig = None, ax = None, pos = 111, **kwargs):
        # if int(str(pos)[0])*int(str(pos)[1]) >= 10 and len(str(pos)) == 3:
        #     pos = int(str(pos)[:2] + "0" + str(pos)[2])
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(pos, title=kwargs.get("title"))
        
        value = None
        if self.magnitude:
            value = self.measurements[3]
            # s = ax.scatter(self.measurements[0], self.measurements[1], c=self.measurements[3], cmap='viridis', vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
        else:
            if kwargs.get("component") is None:
                value = np.linalg.norm(self.measurements[3:], axis=0)

                # s = ax.scatter(self.measurements[0], self.measurements[1], c=magnitude, cmap='viridis', vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))
            else:
                value = self.measurements[3 + kwargs.get("component")]
                # s = ax.scatter(self.measurements[0], self.measurements[1], c=self.measurements[3 + kwargs.get("component")], cmap='viridis', vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"))

        vmin = kwargs.get("vmin")
        vmax = kwargs.get("vmax")
        s = ax.scatter(self.measurements[0], self.measurements[1], c=value, cmap='viridis', vmin=vmin, vmax=vmax)
    
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        if kwargs.get("hide_colorbar") is None:
            cb = fig.colorbar(s)
        # cbar.set_label("B [nT]")
        
        if kwargs.get("hide_x"):
            ax.set_xticks([])
        if kwargs.get("hide_y"):
            ax.set_yticks([])

        if vmin is None:
            vmin = np.min(value)
        if vmax is None:
            vmax = np.max(value)

        return vmin, vmax, s

    def mean(self):
        """
        Return the mean of the measurements.
        """
        
        return np.mean(self.measurements[3:])
        


class ScalarMeasurement(Measurement):
    def __init__(self, measurements, **kwargs):
        super().__init__(measurements, magnitude=True, **kwargs)

class VectorMeasurement(Measurement):
    def __init__(self, measurements, **kwargs):
        super().__init__(measurements, magnitude=False, **kwargs)