import numpy as np
from magnetic_classes import ScalarMeasurement, VectorMeasurement, Source
from scipy.signal import lfilter
from scipy.linalg import toeplitz

class CorrelatedNoise(Source):
    def __init__(self, c_1: float = 0.01, c_2: float = 0.04, c_3: float = 0.9, filter = [1, -1], seed: int = None):
        """
            c_1: a constant term added to the random noise
            c_2: a constant term multiplied to the random noise
            c_3: the correlation value used to generate the Toeplitz matrix
            filter: the filter used to filter the noise
            seed: the seed used to generate the noise
        """
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3
        self.filter = filter

        if seed is None:
            seed = np.random.randint(0, 2 ** 16 - 1)
        self.seed = seed
        np.random.seed(seed)
    
    def __call__(self, x, y, z, i=0, dt=1, magnitude=False):

        if magnitude == True:
            noise = self.generate(x.shape[0], i.shape[0]).flatten()
            t, x, y, z, _, _ = self.tile_t(i*dt, x, y, z)
            return ScalarMeasurement(np.array([x, y, z, t, noise]), t=i*dt)
        Bx = self.generate(x.shape[0], i.shape[0]).flatten()
        By = self.generate(x.shape[0], i.shape[0]).flatten()
        Bz = self.generate(x.shape[0], i.shape[0]).flatten()

        t, x, y, z, _, _ = self.tile_t(i*dt, x, y, z)
        return VectorMeasurement(np.array([x, y, z, t, Bx, By, Bz]), t=i*dt)


    def generate(self, Nspace, Ntime):
        noise = self.c_1 + self.c_2 * np.random.rand(1, 1)
        noise = noise * np.random.randn(Ntime, Nspace)
        noise = lfilter([1], self.filter, noise, axis=0)
        corr = self.c_3 * np.ones(Nspace)
        corr[0] = 1
        A = np.linalg.cholesky(toeplitz(corr))
        noise = np.dot(A, noise.T)
        return noise
    
    def getParameters(self):
        """
        :return: a dictionary with the parameters of the noise
        """
        return {
            "c_1": self.c_1,
            "c_2": self.c_2,
            "c_3": self.c_3,
            "filter": self.filter,
            "seed": self.seed
        }

if __name__ == '__main__':
    n = 100
    sample_rate = 1
    noise = CorrelatedNoise()
    i = np.arange(n * sample_rate)
    t = i / sample_rate
    measurements = noise(np.array([1,2,3,4]), np.array([1,2,3,4]), np.array([1,2,3,4]), i, 1/sample_rate, magnitude=False).values
    print(measurements.shape)
    measurements = measurements.reshape((3, 4, n*sample_rate))

    from matplotlib import pyplot as plt
    plt.plot(t, measurements[0, 0])
    plt.plot(t, measurements[0, 1])
    plt.plot(t, measurements[0, 2])
    plt.plot(t, measurements[0, 3])
    plt.show()
