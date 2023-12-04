import numpy as np
from magnetic_classes import ScalarMeasurement, VectorMeasurement, Source

class ArMaNoise(Source):
    def __init__(self, ar_coeffs: list, ma_coeffs: list, mu=0, sigma=1, seed=None):
        """
        Create a function source.

        :param expression: a string with the expression of the function
        """
        self.ar_coeffs = ar_coeffs
        self.ma_coeffs = ma_coeffs
        self.mu = mu
        self.sigma = sigma

        if seed is None:
            seed = np.random.randint(0, 2 ** 16 - 1)
        self.seed = seed
        np.random.seed(seed)
    
    def __call__(self, x, y, z, i=0, dt=1, magnitude=False):
        p, q = len(self.ar_coeffs), len(self.ma_coeffs)
        max_order = max(p, q)
        i = np.array([i]).flatten()
        num_samples = i.shape[0]

        if magnitude:
            white_noise = np.random.normal(size=(num_samples + max_order))

            arma = np.zeros((num_samples + max_order))

            # Generate AR and MA terms
            for j in range(max_order, num_samples + max_order):
                ar_term = np.sum(self.ar_coeffs * arma[j-p:j])
                ma_term = np.sum(self.ma_coeffs * white_noise[j-q:j])
                arma[j] = ar_term + ma_term + white_noise[j]

            # Remove the first max_order samples
            arma = arma[max_order:]

            arma = self.mu + self.sigma * arma

            # Combine AR and MA terms to get ARMA noise
            len_x = np.array(x).shape[0]
            x = np.array(x).repeat(num_samples)
            y = np.array(y).repeat(num_samples)
            z = np.array(z).repeat(num_samples)
            arma = np.tile(arma, len_x)
            t = np.tile(i*dt, len_x)

            return ScalarMeasurement(np.array([x, y, z, t, arma]))
        
        else:
            white_noise = np.random.normal(size=(num_samples + max_order, 3))

            arma = np.zeros((num_samples + max_order, 3))

            # Generate AR and MA terms
            for j in range(max_order, num_samples + max_order):
                ar_term = np.dot(self.ar_coeffs, arma[j-p:j]) 
                ma_term = np.dot(self.ma_coeffs, white_noise[j-q:j])
                arma[j] = ar_term + ma_term + white_noise[j]
            
            # Remove the first max_order samples
            arma = arma[max_order:]

            arma = self.mu + self.sigma * arma

            # Combine AR and MA terms to get ARMA noise
            len_x = np.array(x).shape[0]
            x = np.array(x).repeat(num_samples)
            y = np.array(y).repeat(num_samples)
            z = np.array(z).repeat(num_samples)
            arma = np.tile(arma, (len_x, 1))
            t = np.tile(i*dt, len_x)

            return VectorMeasurement(np.array([x, y, z, t, arma[:, 0], arma[:, 1], arma[:, 2]]))
    
    def getParameters(self):
        """
        :return: a dictionary with the parameters of the noise
        """
        return {
            "ar_coeffs": self.ar_coeffs,
            "ma_coeffs": self.ma_coeffs,
            "mu": self.mu,
            "sigma": self.sigma,
            "seed": self.seed
        }

if __name__ == '__main__':
    n = 100
    sample_rate = 5
    ma_coeffs = np.random.normal(size=5*sample_rate)
    noise = ArMaNoise(ar_coeffs=[0.9], ma_coeffs=ma_coeffs, mu=1e3, sigma=10)
    t = np.linspace(0, n, n*sample_rate)
    measurements = noise(np.array([1,2,3]), np.array([1,2,3]), np.array([1,2,3]), t, magnitude=False).values
    print(measurements.shape)
    measurements = measurements.reshape((3, 3, n*sample_rate))

    from matplotlib import pyplot as plt
    plt.plot(t, measurements[0, 0])
    plt.plot(t, measurements[0, 1])
    plt.plot(t, measurements[0, 2])
    plt.show()
