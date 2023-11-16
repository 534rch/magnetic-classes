import numpy as np

from .field import Field

class Layer(Field):
    def __init__(self, dx, dy, sdz, nx:int, ny:int, depth, seed=None, dist='normal', I=1.0):
        """
        Create a layer of dipoles. A layer is similar to a grid but has a random z-coordinate.
        :param dx: x-spacing of the grid.
        :param dy: y-spacing of the grid.
        :param sdz: standard deviation of the z-coordinate of the dipoles.
        :param nx: number of x-points in the grid. (in total 2*nx+1)
        :param ny: number of y-points in the grid. (in total 2*ny+1)
        :param depth: depth of the layer.
        :param seed: random seed.
        :param dist: distribution of the z-coordinate of the dipoles. Can be 'normal', 'uniform' or 'triangular'.
        :param I: intensity of the dipoles in nT at the surface.
        """
        super().__init__(seed)

        self.dx = dx
        self.dy = dy
        self.sdz = sdz
        self.nx = nx
        self.ny = ny
        self.depth = depth
        self.dist = dist
        self.I = I

        self.x = np.linspace(-nx*dx, nx*dx, 2*nx+1)
        self.y = np.linspace(-ny*dy, ny*dy, 2*ny+1)
        
        xy = np.meshgrid(self.x, self.y, indexing='ij')

        # Set random seed
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        if dist == 'normal':
            # z-coordinate of the dipoles is the depth taken from a normal distribution with depth [depth] and standard deviation sdz
            self.z = np.random.normal(depth, sdz, size=xy[0].shape)
        elif dist == 'uniform':
            # Distribute the dipoles uniformly from [depth - sdz, depth + sdz]
            self.z = np.random.uniform(depth-sdz, depth+sdz, size=xy[0].shape)
        elif dist == 'triangular':
            # Distribute the number of dipoles in a trangle: more dipoles at depth + sdz, less at depth - sdz.
            self.z = np.random.triangular(depth-sdz, depth+sdz, depth+sdz, size=xy[0].shape)
        self.xyz = np.dstack((xy[0], xy[1], self.z))
        self.xyz = self.xyz.reshape((-1, 3))
        self.dipoles = self.xyz.T

        self.M = np.random.rand(3* (2*nx+1) * (2*ny+1)).reshape((-1,3)) * I
    
    def getParameters(self):
        """
        :return: a dictionary with the parameters of the layer
        """

        return {
            "dx": self.dx,
            "dy": self.dy,
            "sdz": self.sdz,
            "nx": self.nx,
            "ny": self.ny,
            "depth": self.depth,
            "dist": self.dist,
            "I": self.I
        }

if __name__ == "__main__":
    shallowlayer = Layer(40, 40, 10, 50, 50, -600)
    deeplayer = Layer(1000, 1000, 0, 10, 10, -6000)

    print('shallowlayer: ', shallowlayer.xyz.shape)
    print('deeplayer: ', deeplayer.xyz.shape)

    # 3D plot of the dipoles
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(shallowlayer.xyz[:, 0], shallowlayer.xyz[:, 1], shallowlayer.xyz[:, 2], c='b', marker='o', label='Shallow layer')
    ax.scatter(deeplayer.xyz[:, 0], deeplayer.xyz[:, 1], deeplayer.xyz[:, 2], c='r', marker='o', label='Deep layer')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


    

    