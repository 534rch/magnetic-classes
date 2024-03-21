import numpy as np
from .source import Source
from .dipole import Dipole
from ..paths.path import Path
from .measurement import ScalarMeasurement, VectorMeasurement


def rotation_matrix(dx, dy, dz):
    # Normalize the vector
    v = np.array([dx, dy, dz])
    # print("v", v.shape)
    # print(np.linalg.norm(v, axis=1))
    v /= np.linalg.norm(v)

    # Calculate the angle between [1, 0, 0] and the desired vector
    angle = np.arccos(np.dot([1, 0, 0], v))

    # Calculate the axis of rotation
    axis = np.cross([1, 0, 0], v)
    if np.linalg.norm(axis) == 0:
        return np.eye(3)
    axis /= np.linalg.norm(axis)

    # Rodrigues' rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    rotation_matrix = np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )
    return rotation_matrix


def rotation_matrix_vector(
    dx: np.ndarray, dy: np.ndarray, dz: np.ndarray
) -> np.ndarray:
    # Check if shapes are the same
    assert dx.shape == dy.shape == dz.shape

    rotation_matrices = []
    for i in range(len(dx)):
        rotation_matrices.append(rotation_matrix(dx[i], dy[i], dz[i]))

    return np.array(rotation_matrices)


class MovingDipole(Source):
    def __init__(self, dipole: Dipole, path: Path):
        self.dipole = dipole
        self.path = path

    def __call__(self, x, y, z, i, dt=1, magnitude=False):
        """
        Calculate the magnetic field at a given point.
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :param z: z-coordinate of the point
        :param i: time index
        :param dt: time step
        :return: the magnetic field strength at the point
        """

        # Moving the dipole is the same as fixing the dipole and moving the point
        xt, yt, zt = self.path(i)

        len_x = np.array(x).shape[0]
        len_t = np.array(xt).shape[0]
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        dx, dy, dz = self.path.derivative()

        measurements = np.zeros((len_x, len(i), 7))

        t = i * dt

        for k in range(len(i)):
            R = rotation_matrix(dx[k], dy[k], dz[k])

            mx, my, mz = self.dipole.mx, self.dipole.my, self.dipole.mz
            m = np.array([mx, my, mz])
            m = np.dot(R, m)
            mx, my, mz = m

            # Permeability of free space
            mu0 = 4 * np.pi * 10 ** (-7)  # N/A^2 or H/m

            # r : from with center of the dipole to the location
            rx = x - xt[k]
            ry = y - yt[k]
            rz = z - zt[k]
            r = np.sqrt(rx**2 + ry**2 + rz**2)

            r_in_m = rx * mx + ry * my + rz * mz

            # Do not divide by r if grid point is at same location as dipole
            r5 = r**5
            r_in_m_5 = np.divide(
                r_in_m, r5, out=np.zeros_like(r5), where=r5 != 0, casting="unsafe"
            )
            r3 = r**3

            r_3 = np.divide(
                1, r3, out=np.zeros_like(r3), where=r3 != 0, casting="unsafe"
            )

            Bx = mu0 / (4 * np.pi) * (3 * r_in_m_5 * rx - mx * r_3)
            By = mu0 / (4 * np.pi) * (3 * r_in_m_5 * ry - my * r_3)
            Bz = mu0 / (4 * np.pi) * (3 * r_in_m_5 * rz - mz * r_3)

            tk = np.array(t[k]).repeat(len_x)
            measurements[:, k, :] = np.array([x, y, z, tk, Bx, By, Bz]).T

        # Flatten the measurements
        measurements = measurements.reshape((len(i) * len_x, 7))

        if magnitude:
            measurements[:, 4] = np.linalg.norm(measurements[:, 4:], axis=1)
            measurements = measurements[:, :5]

            return ScalarMeasurement(measurements.T, t=i * dt)

        return VectorMeasurement(measurements.T, t=i * dt)

    def getParameters(self):
        return {}
