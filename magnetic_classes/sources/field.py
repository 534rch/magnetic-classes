import numpy as np
from .dipole import Dipole
# from .layer import Layer
from .source import Source, parallel_source
from .measurement import Measurement, ScalarMeasurement, VectorMeasurement
import json
import logging
import copy

import multiprocessing as mp

class Field(Source):
    """
    Abstract class for a magnetic fields that are composed of sources (e.g. dipoles)
    """

    def __init__(self, seed=None):
        self.sources = []

        if seed is None:
            seed = np.random.randint(0, 2 ** 16 - 1)
        self.seed = seed
        np.random.seed(seed)

        self.M = np.random.rand(3).reshape((1, 3))
        self.dipoles = None

    def placeDipoles(self):
        """
        Place dipoles in the grid
        """
        if self.dipoles is None:
            raise Exception("No dipoles defined.")

        for i in range(self.dipoles.shape[1]):
            if self.M.shape == (1,3):
                M = copy.deepcopy(self.M[0,:])
            else:
                M = copy.deepcopy(self.M[i,:])

            # # Assert that M is close to a unit vector
            # assert np.isclose(np.linalg.norm(M), 1), "M is not a unit vector"

            # Scale magnetic moment cubically with the depth of the dipole (z).

            # Print the type of M
            M *= np.abs(self.dipoles[2, i]) ** 3 * 1e7/2

            self.sources.append(
                Dipole(
                    self.dipoles[0, i],
                    self.dipoles[1, i],
                    self.dipoles[2, i],
                    M[0],
                    M[1],
                    M[2],
                    fit_to_surface=False,
                )
            )

    def __call__(self, x, y, z, i=0, dt=1, magnitude=False, parallel=False) -> Measurement:
        """
        :param x: x-coordinate of the point
        :param y: y-coordinate of the point
        :param z: z-coordinate of the point
        :param i: time index
        :param dt: time step
        :return: the magnetic field at the point
        """
        if not self.sources and self.dipoles is not None:
            self.placeDipoles()

        if parallel:
            with mp.Pool(processes=len(self.sources)) as pool:
                results = [pool.apply_async(parallel_source, args=(source, x, y, z, i, dt)) for source in self.sources]
                B = [result.get() for result in results]
            Bx, By, Bz = np.sum(B, axis=0)
        else:
            # Sum all nd arrays along all axes
            # print(np.sum([source(x, y, z) for source in self.sources], axis=
            Bx, By, Bz = np.sum([source(x, y, z, i, dt).values for source in self.sources], axis=0)
            # Bx, By, Bz = np.sum([source(x, y, z) for source in self.sources], axis=0)

        if magnitude:
            return ScalarMeasurement(np.array([x, y, z, np.linalg.norm([Bx, By, Bz], axis=0)]))

        return VectorMeasurement(np.array([x, y, z, Bx, By, Bz]))

    def getParameters(self):
        """
        :return: a dictionary with the parameters of the grid
        """
        data = []
        for source in self.sources:
            item = {
                "type": source.__class__.__name__,
                "parameters": source.getParameters(),
            }
            data.append(item)

        return data

    def export(self, path=None, description=None):
        """
        Export the field to a json file
        :param path: path to the json file

        The json file will have the following structure:
        {
            "type": "grid",
            "parameters": {
                // Field parameters
            }
        }
        """
        if self.seed is None:
            raise ValueError('Cannot export parameters without a seed.')
        
        data = {
            "type": self.__class__.__name__,
            "description": description,
            "parameters": self.getParameters(),
            "seed": self.seed,
        }

        if path is not None:
            with open(path, "w") as f:
                # Dump nicely formatted json
                json.dump(data, f, indent=4)

            print("Exported field to {}".format(path))

        return data
    
    def __add__(self, other):
        """
            Add two fields together
        """

        if not isinstance(other, Field):
            raise TypeError("Can only add two fields together.")
        
        # Show a warning in the console that combining adding fields breaks the seed
        logging.warning("Adding two fields together will break the seed.")

        # Create a new field
        new_field = Field()
        new_field.seed = None

        # Add all sources
        new_field.sources = [self, other]
        # new_field.M = np.concatenate((self.M, other.M), axis=0)
        
        # new_field.dipoles = []
        # if self.dipoles is not None:
        #     new_field.dipoles.append(self.dipoles)
        # if other.dipoles is not None:
        #     new_field.dipoles.append(other.dipoles)

        return new_field

    def plot(self, **kwargs):
        if len(self.sources) == 0:
            raise Exception("No sources defined.")

        # Plot all sources in 3D
        from matplotlib import pyplot as plt
        import matplotlib
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if "title" in kwargs:
            ax.set_title(kwargs["title"])

        min_m = np.inf
        max_m = 0

        from .layer import Layer
        for source in self.sources:
            if isinstance(source, Dipole):
                m = np.linalg.norm([source.mx, source.my, source.mz]) / (np.abs(source.z0)**3  * 1e7/2)
                min_m = min(min_m, m)
                max_m = max(max_m, m)
            if isinstance(source, Layer):
                print(source.M.shape, source.dipoles.shape)
                M = np.linalg.norm(source.M, axis=1)
                # / np.abs(source.dipoles[2, :])**3
                min_m = min(min_m, np.min(M))
                max_m = max(max_m, np.max(M))
                # for dipole in source.dipoles:
                #     m = dipole.
                #     m = np.linalg.norm([dipole.mx, dipole.my, dipole.mz]) / np.abs(dipole.z0)**3
                #     min_m = min(min_m, m)
                #     max_m = max(max_m, m)

        norm = matplotlib.colors.Normalize(vmin=min_m, vmax=max_m)
        # Color all dipoles red
        for source in self.sources:
            if isinstance(source, Dipole):
                m = np.linalg.norm([source.mx, source.my, source.mz]) / (np.abs(source.z0)**3  * 1e7/2)

                if m < 0.01*max_m:
                    continue
                color = (m - min_m) / (max_m - min_m)
                ax.scatter(source.x0, source.y0, source.z0, c=plt.cm.viridis(norm(m)))
            if isinstance(source, Layer):
                source.placeDipoles()
                for dipole in source.sources:
                    m = np.linalg.norm([dipole.mx, dipole.my, dipole.mz]) / (np.abs(dipole.z0)**3 * 1e7/2)

                    if m < 0.01*max_m:
                        continue
                    color = (m - min_m) / (max_m - min_m)
                    ax.scatter(dipole.x0, dipole.y0, dipole.z0, c=plt.cm.viridis(norm(m)))

        ax.legend(["Dipole"])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        # Plot the colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        
        # ax.legend(["Dipole", "Magnetic moment"])
        return fig, ax

def importField(path):
    """
    Import a field from a json file
    :param path: path to the json file
    :return: the field
    """
    
    # Add .json extension if not present
    if not path.endswith('.json'):
        path += '.json'
        
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        try:
            # Search in the sample fields directory "./samples/fields/"
            import os
            with open(os.path.join(os.path.dirname(__file__)+'/../', "samples/fields", path), "r") as f:
                data = json.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("Could not find file {}".format(path))

    if data["type"] == "DipoleGrid":
        from .dipole_grid import DipoleGrid

        return DipoleGrid(**data["parameters"])
    if data["type"] == "Layer":
        from .layer import Layer

        return Layer(**data["parameters"], seed=data["seed"])
    elif data["type"] == "Field":
        field = Field()
        for source in data["parameters"]:
            if source["type"] == "Dipole":
                from .dipole import Dipole
                field.sources.append(
                    Dipole(**source["parameters"])
                )
            elif source["type"] == "DipoleGrid":
                from .dipole_grid import DipoleGrid
                field.sources.append(
                    DipoleGrid(**source["parameters"])
                )
            elif source["type"] == "Layer":
                from .layer import Layer
                field.sources.append(
                    Layer(**source["parameters"], seed=data["seed"])
                )
            elif source["type"] == "GaussianNoise":
                from .noises.gaussian import GaussianNoise
                field.sources.append(
                    GaussianNoise(**source["parameters"])
                )
            else:
                raise Exception("Unknown source type: {}".format(source["type"]))

        return field
    else:
        raise Exception("Unknown field type: {}".format(data["type"]))
