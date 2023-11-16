import numpy as np

class Path():
    def __init__(self, x = None, y = None, z = None, **kwargs):
        assert len(x) == len(y) == len(z), "x, y and z must have the same length"
        self.x = x
        self.y = y
        self.z = z


        if kwargs.get("grid") is not None:
            self.grid : np.ndarray = kwargs.get("grid")

    def __call__(self, t = None):
        if t is None:
            return self.x, self.y, self.z
        else:
            return self.x[t], self.y[t], self.z[t]
    
    def rotate(self, theta, phi):
        # Rotate the path
        self.x = self.x * np.cos(theta) - self.y * np.sin(theta)
        self.y = self.x * np.sin(theta) + self.y * np.cos(theta)
        self.z = self.z * np.cos(phi) - self.y * np.sin(phi)
        self.y = self.z * np.sin(phi) + self.y * np.cos(phi)

        print("Warning: Path.rotate() is not implemented for grid.")
        return self

    def translate(self, x, y, z):
        # Translate the path
        self.x += x
        self.y += y
        self.z += z

        if hasattr(self, 'grid'):
            self.grid[0] += x
            self.grid[1] += y
            self.grid[2] += z
        return self

    def center(self):
        # Center the path
        self.translate(-np.mean(self.x), -np.mean(self.y), -np.mean(self.z))
        return self
        
    def plot(self, scatter: bool = False):
        # Plot the path
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if scatter:
            ax.scatter(self.x, self.y, self.z, label='Points')
        else:
            ax.plot(self.x, self.y, self.z, label='Path')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title(r'Trajectory')
        ax.legend()
        plt.show()