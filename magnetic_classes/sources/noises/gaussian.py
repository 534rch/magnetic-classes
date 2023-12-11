import numpy as np
from magnetic_classes import ScalarMeasurement, VectorMeasurement, Source

class GaussianNoise(Source):
    def __init__(self, mean: float = 0, std: float = 1, seed: int = None):
        """
        mean: float
            Mean of the Gaussian distribution
        std: float
            Standard deviation of the Gaussian distribution
        """
        self.mean = mean
        self.std = std
        if seed is None:
            seed = np.random.randint(0, 2 ** 16 - 1)
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, x, y, z, i=0, dt=1, magnitude=False):
        """
        Generate Gaussian noise
        """
        t = i * dt
        np.random.seed(self.seed+t)
        
        t, x, y, z, _, _ = self.tile_t(t, x, y, z)

        if magnitude:
            noise = np.random.normal(self.mean, self.std, size=x.shape)
            return ScalarMeasurement(np.array([x, y, z, t, noise]), t=i*dt)
        else:
            noise = np.random.normal(self.mean, self.std, size=(x.shape[0], 3))
            return VectorMeasurement(np.array([x, y, z, t, noise[:, 0], noise[:, 1], noise[:, 2]]), t=i*dt)
    
    def getParameters(self):
        """
        :return: a dictionary with the parameters of the noise
        """
        return {
            "mean": self.mean,
            "std": self.std,
            "seed": self.seed
        }

if __name__ == '__main__':
    noise = GaussianNoise()
    print(noise(0, 0, 0, magnitude=True).values)