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
            seed = np.random.randint(0, 2 ** 32 - 1)
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, x, y, z, t=0, magnitude=False):
        """
        Generate Gaussian noise
        """
        np.random.seed(self.seed+t)
        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        if magnitude:
            noise = np.random.normal(self.mean, self.std, size=x.shape)
            return ScalarMeasurement(np.array([x, y, z, noise]))
        else:
            noise = np.random.normal(self.mean, self.std, size=(x.shape[0], 3))
            return VectorMeasurement(np.array([x, y, z, noise[:, 0], noise[:, 1], noise[:, 2]]))
    
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