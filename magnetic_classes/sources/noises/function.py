import numpy as np
from magnetic_classes import ScalarMeasurement, VectorMeasurement, Source

class Function(Source):
    allowed_variables = ["x", "y", "z", "t", "np", "pi", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "exp", "log", "log10", "sqrt", "abs"]

    def __init__(self, expression_x = "0", expression_y = "0", expression_z = "0", parameters = {}):
        """
        Create a function source.

        :param expression: a string with the expression of the function
        """
        self.raw_expression_x = expression_x
        self.raw_expression_y = expression_y
        self.raw_expression_z = expression_z
        self.parameters = parameters

        # Replace the parameters in the expression
        for key, value in parameters.items():
            expression_x = expression_x.replace(key, str(value))
            expression_y = expression_y.replace(key, str(value))
            expression_z = expression_z.replace(key, str(value))

        for x in [expression_x, expression_y, expression_z]:
            res = x
            for v in self.allowed_variables:
                res = res.replace(v, " ")
            for letter in res:
                if letter.isalpha():
                    raise ValueError("Invalid variable in expression_{}: {}".format(x, letter))
        
        self.expression_x = expression_x
        self.expression_y = expression_y
        self.expression_z = expression_z

    def __call__(self, x, y, z, t=0, magnitude=False):
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        if not isinstance(z, np.ndarray):
            z = np.array([z])
        if not isinstance(t, np.ndarray):
            t = np.array([t])
            
        len_x = np.array(x).shape[0]
        x = np.array(x).repeat(t.shape[0])
        y = np.array(y).repeat(t.shape[0])
        z = np.array(z).repeat(t.shape[0])
        t = np.tile(t, len_x)

        # Evaluate the expression
        res = np.zeros((x.shape[0], 3))
        res_x = eval(self.expression_x)
        res_y = eval(self.expression_y)
        res_z = eval(self.expression_z)
        res[:, 0] = res_x
        res[:, 1] = res_y
        res[:, 2] = res_z

        if magnitude:
            return ScalarMeasurement(np.array([x, y, z, t, np.linalg.norm(res, axis=1)]))
        else:
            return VectorMeasurement(np.array([x, y, z, t, res[:, 0], res[:, 1], res[:, 2]]))
    
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
    noise = Function("x**2 + y**2 + z**2 + t", "t")
    measurements = noise(np.array([1,2,3]), np.array([1,2,3]), np.array([1,2,3]), [1,2,3,4,5], magnitude=True)
    print(measurements)