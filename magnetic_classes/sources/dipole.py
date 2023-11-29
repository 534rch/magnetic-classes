import numpy as np
from .source import Source
from .measurement import ScalarMeasurement, VectorMeasurement


class Dipole(Source):
    """
    Initialize a dipole object.
    location of dipole
    :param x0: x-coordinate of the dipole
    :param y0: y-coordinate of the dipole
    :param z0: z-coordinate of the dipole
    magnetic dipole moment
    :param mx: x-coordinate of the magnetic dipole moment
    :param my: y-coordinate of the magnetic dipole moment
    :param mz: z-coordinate of the magnetic dipole moment
    """

    def __init__(self, x0=0, y0=0, z0=0, mx=0, my=0, mz=0, active=True, fit_to_surface=False):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.mx = mx
        self.my = my
        self.mz = mz
        self.active = active
        self.fit_to_surface = fit_to_surface

        if fit_to_surface:
            self.mx *= np.abs(z0) ** 3 * 1e7/2
            self.my *= np.abs(z0) ** 3 * 1e7/2
            self.mz *= np.abs(z0) ** 3 * 1e7/2

    def __call__(self, x, y, z, t=0, magnitude=False):
        """
        Calculate the magnetic field at a given point.
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :param z: z-coordinate of the point
        :return: the magnetic field strength at the point
        """
        # Check if value is not a numpy array
        if not isinstance(t, np.ndarray) or t.shape[0] == 1:
            t = np.array([t]).repeat(x.shape[0])

    

        if not self.active:
            if magnitude:
                return ScalarMeasurement(np.array([x,y,z,t, np.zeros_like(x)]))
            else:
                return VectorMeasurement(np.array([x,y,z,t, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]))

        # Fundamental constants

        # Permeability of free space
        mu0 = 4 * np.pi * 10 ** (-7)  # N/A^2 or H/m

        # r : from with center of the dipole to the location
        rx = x - self.x0
        ry = y - self.y0
        rz = z - self.z0
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        r_in_m = rx * self.mx + ry * self.my + rz * self.mz

        # Do not divide by r if grid point is at same location as dipole
        r5 = r**5
        r_in_m_5 = np.divide(
            r_in_m, r5, out=np.zeros_like(r5), where=r5 != 0, casting="unsafe"
        )
        r3 = r**3

        r_3 = np.divide(1, r3, out=np.zeros_like(r3), where=r3 != 0, casting="unsafe")

        Bx = mu0 / (4 * np.pi) * (3 * r_in_m_5 * rx - self.mx * r_3)
        By = mu0 / (4 * np.pi) * (3 * r_in_m_5 * ry - self.my * r_3)
        Bz = mu0 / (4 * np.pi) * (3 * r_in_m_5 * rz - self.mz * r_3)

        if magnitude:
            return ScalarMeasurement(np.array([x, y, z, t, np.linalg.norm([Bx, By, Bz], axis=0)]))

        return VectorMeasurement(np.array([x, y, z, t, Bx, By, Bz]))

    def moment(self):
        """
        :return: dipole moment in np.array([mx,my,mz])
        """

        return np.array([self.mx, self.my, self.mz])

    def loc(self):
        """
        :return: the spacial location of the dipole in np.array([x,y,z])
        """
        return np.array([self.x0, self.y0, self.z0])

    def enable(self):
        """
        Enable the dipole
        """
        self.active = True

    def disable(self):
        """
        Disable the dipole
        """
        self.active = False

    def getParameters(self):
        """
        :return: a dictionary with the parameters of the dipole
        """
        return {
            "x0": self.x0,
            "y0": self.y0,
            "z0": self.z0,
            "mx": self.mx,
            "my": self.my,
            "mz": self.mz,
            "active": self.active,
            "fit_to_surface": self.fit_to_surface
        }
