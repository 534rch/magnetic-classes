import numpy as np
from .source import Source
from .measurement import ScalarMeasurement, VectorMeasurement


class Constant(Source):
    def __init__(self, Bx=0, By=0, Bz=0, active=True):
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.active = active

    def __call__(self, x, y, z, i, dt=1, magnitude=False):
        """
        Calculate the magnetic field at a given point.
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :param z: z-coordinate of the point
        :return: the magnetic field strength at the point
        """
        t = i * dt

        # Note: a dipole is not time-dependent, so Bx, By, Bz are just repeated for each time step

        if not self.active:
            t, x, y, z, _, _ = self.tile_t(t, x, y, z)
            if magnitude:
                return ScalarMeasurement(
                    np.array([x, y, z, t, np.zeros_like(x)]), t=i * dt
                )
            else:
                return VectorMeasurement(
                    np.array(
                        [
                            x,
                            y,
                            z,
                            t,
                            np.zeros_like(x),
                            np.zeros_like(x),
                            np.zeros_like(x),
                        ]
                    ),
                    t=i * dt,
                )

        t, x, y, z, _, len_t = self.tile_t(t, x, y, z)
        Bx = np.array(self.Bx) * np.ones_like(x)
        By = np.array(self.By) * np.ones_like(x)
        Bz = np.array(self.Bz) * np.ones_like(x)

        if magnitude:
            return ScalarMeasurement(
                np.array([x, y, z, t, np.linalg.norm([Bx, By, Bz], axis=0)]), t=i * dt
            )

        return VectorMeasurement(np.array([x, y, z, t, Bx, By, Bz]), t=i * dt)

    def getParameters(self):
        """
        :return: a dictionary with the parameters of the dipole
        """
        return {
            "Bx": self.Bx,
            "By": self.By,
            "Bz": self.Bz,
            "active": self.active,
        }
