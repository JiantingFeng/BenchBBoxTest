import numpy as np
from typing import Dict, Tuple, Optional, Callable, Union, List
from ...core import DataGenerator


class PostNonlinearGenerator(DataGenerator):
    """
    Post-nonlinear models from Doran et al. (2014).

    Implements data generation for post-nonlinear models under both
    null and alternative hypotheses. The models follow the structure:

    Under null hypothesis (X ⊥ Y | Z):
    - Z ~ N(0, I_p)
    - X = f(Z^T a_x + ε_x), where ε_x ~ N(0, σ²)
    - Y = f(Z^T a_y + ε_y), where ε_y ~ N(0, σ²)

    Under alternative hypothesis (X ⊥̸ Y | Z):
    - Z ~ N(0, I_p)
    - X = f(Z^T a_x + ε_x), where ε_x ~ N(0, σ²)
    - Y = f(Z^T a_y + b * X + ε_y), where ε_y ~ N(0, σ²)

    where:
    - p is the dimension of Z
    - a_x, a_y are coefficient vectors
    - b controls the strength of dependence between X and Y
    - σ² is the noise variance
    - f is a nonlinear function (e.g., tanh, sigmoid)
    """

    # Dictionary of available nonlinear functions
    NONLINEAR_FUNCTIONS = {
        "tanh": np.tanh,
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "relu": lambda x: np.maximum(0, x),
        "cubic": lambda x: x**3,
        "identity": lambda x: x,
    }

    def __init__(
        self,
        p: int = 5,
        a_x: Optional[np.ndarray] = None,
        a_y: Optional[np.ndarray] = None,
        nonlinear_func: Union[str, Callable] = "tanh",
        covariance_z: Optional[np.ndarray] = None,
    ):
        """
        Initialize the post-nonlinear generator.

        Args:
            p: Dimension of the conditioning variable Z
            a_x: Coefficients for X ~ Z relationship (if None, random values are generated)
            a_y: Coefficients for Y ~ Z relationship (if None, random values are generated)
            nonlinear_func: Nonlinear function to apply ('tanh', 'sigmoid', 'relu', 'cubic', 'identity') or a callable
            covariance_z: Covariance matrix for Z (if None, identity matrix is used)
        """
        self.p = p

        # Set nonlinear function
        if isinstance(nonlinear_func, str):
            if nonlinear_func in self.NONLINEAR_FUNCTIONS:
                self.nonlinear_func = self.NONLINEAR_FUNCTIONS[nonlinear_func]
            else:
                raise ValueError(
                    f"Unknown nonlinear function: {nonlinear_func}. "
                    f"Available options are: {list(self.NONLINEAR_FUNCTIONS.keys())}"
                )
        elif callable(nonlinear_func):
            self.nonlinear_func = nonlinear_func
        else:
            raise ValueError("nonlinear_func must be a string or a callable")

        # Initialize coefficients if not provided
        if a_x is None:
            self.a_x = np.random.normal(0, 1, size=p)
        else:
            self.a_x = a_x

        if a_y is None:
            self.a_y = np.random.normal(0, 1, size=p)
        else:
            self.a_y = a_y

        # Initialize covariance matrix for Z
        if covariance_z is None:
            self.covariance_z = np.eye(p)  # Identity matrix
        else:
            # Ensure the covariance matrix is valid (symmetric and positive semi-definite)
            if covariance_z.shape != (p, p):
                raise ValueError(f"Covariance matrix must have shape ({p}, {p})")
            self.covariance_z = covariance_z

    def generate_null(
        self, n_samples: int, noise_scale: float = 0.5, return_params: bool = False
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Generate data under the null hypothesis (X ⊥ Y | Z).

        Args:
            n_samples: Number of samples to generate
            noise_scale: Scale of the noise (standard deviation)
            return_params: Whether to return the parameters used for generation

        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays, and optionally 'params'
        """
        # Generate Z ~ N(0, Σ)
        Z = np.random.multivariate_normal(
            mean=np.zeros(self.p), cov=self.covariance_z, size=n_samples
        )

        # Generate X = f(Z * a_x + epsilon_x)
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X_linear = Z @ self.a_x + epsilon_x
        X = self.nonlinear_func(X_linear)

        # Generate Y = f(Z * a_y + epsilon_y) (independent of X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y_linear = Z @ self.a_y + epsilon_y
        Y = self.nonlinear_func(Y_linear)

        result = {"X": X.reshape(-1, 1), "Y": Y.reshape(-1, 1), "Z": Z}

        if return_params:
            params = {
                "a_x": self.a_x,
                "a_y": self.a_y,
                "noise_scale": noise_scale,
                "covariance_z": self.covariance_z,
                "nonlinear_func": self.nonlinear_func.__name__
                if hasattr(self.nonlinear_func, "__name__")
                else str(self.nonlinear_func),
                "hypothesis": "null",
            }
            result["params"] = params

        return result

    def generate_alternative(
        self,
        n_samples: int,
        b: float = 0.5,
        noise_scale: float = 0.5,
        return_params: bool = False,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Generate data under the alternative hypothesis (X ⊥̸ Y | Z).

        Args:
            n_samples: Number of samples to generate
            b: Strength of dependence between X and Y
            noise_scale: Scale of the noise (standard deviation)
            return_params: Whether to return the parameters used for generation

        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays, and optionally 'params'
        """
        # Generate Z ~ N(0, Σ)
        Z = np.random.multivariate_normal(
            mean=np.zeros(self.p), cov=self.covariance_z, size=n_samples
        )

        # Generate X = f(Z * a_x + epsilon_x)
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X_linear = Z @ self.a_x + epsilon_x
        X = self.nonlinear_func(X_linear)

        # Generate Y = f(Z * a_y + b * X + epsilon_y) (dependent on X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y_linear = Z @ self.a_y + b * X + epsilon_y
        Y = self.nonlinear_func(Y_linear)

        result = {"X": X.reshape(-1, 1), "Y": Y.reshape(-1, 1), "Z": Z}

        if return_params:
            params = {
                "a_x": self.a_x,
                "a_y": self.a_y,
                "b": b,
                "noise_scale": noise_scale,
                "covariance_z": self.covariance_z,
                "nonlinear_func": self.nonlinear_func.__name__
                if hasattr(self.nonlinear_func, "__name__")
                else str(self.nonlinear_func),
                "hypothesis": "alternative",
            }
            result["params"] = params

        return result

    def generate_dataset(
        self,
        n_samples: int,
        hypothesis: str = "null",
        b: float = 0.5,
        noise_scale: float = 0.5,
        return_params: bool = False,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Generate a dataset under either hypothesis.

        Args:
            n_samples: Number of samples to generate
            hypothesis: 'null' or 'alternative'
            b: Strength of dependence between X and Y (for alternative)
            noise_scale: Scale of the noise
            return_params: Whether to return the parameters used for generation

        Returns:
            Dictionary containing generated data
        """
        if hypothesis.lower() == "null":
            return self.generate_null(n_samples, noise_scale, return_params)
        elif hypothesis.lower() == "alternative":
            return self.generate_alternative(n_samples, b, noise_scale, return_params)
        else:
            raise ValueError("hypothesis must be 'null' or 'alternative'")
