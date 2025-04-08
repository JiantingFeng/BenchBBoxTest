import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from ...core import DataGenerator


class LinearGaussianGenerator(DataGenerator):
    """
    Linear Gaussian models from Shah and Peters (2020).

    Implements data generation for linear Gaussian models under both
    null and alternative hypotheses. The models follow the structure:

    Under null hypothesis (X ⊥ Y | Z):
    - Z ~ N(0, I_d)
    - X = Z^T β_x + ε_x, where ε_x ~ N(0, σ²)
    - Y = Z^T β_y + ε_y, where ε_y ~ N(0, σ²)

    Under alternative hypothesis (X ⊥̸ Y | Z):
    - Z ~ N(0, I_d)
    - X = Z^T β_x + ε_x, where ε_x ~ N(0, σ²)
    - Y = Z^T β_y + γ * X + ε_y, where ε_y ~ N(0, σ²)

    where:
    - d is the dimension of Z
    - β_x, β_y are coefficient vectors
    - γ controls the strength of dependence between X and Y
    - σ² is the noise variance
    """

    def __init__(
        self,
        d: int = 5,
        beta_x: Optional[np.ndarray] = None,
        beta_y: Optional[np.ndarray] = None,
        covariance_z: Optional[np.ndarray] = None,
    ):
        """
        Initialize the linear Gaussian generator.

        Args:
            d: Dimension of the conditioning variable Z
            beta_x: Coefficients for X ~ Z relationship (if None, random values are generated)
            beta_y: Coefficients for Y ~ Z relationship (if None, random values are generated)
            covariance_z: Covariance matrix for Z (if None, identity matrix is used)
        """
        self.d = d

        # Initialize coefficients if not provided
        if beta_x is None:
            self.beta_x = np.random.normal(0, 1, size=d)
        else:
            self.beta_x = beta_x

        if beta_y is None:
            self.beta_y = np.random.normal(0, 1, size=d)
        else:
            self.beta_y = beta_y

        # Initialize covariance matrix for Z
        if covariance_z is None:
            self.covariance_z = np.eye(d)  # Identity matrix
        else:
            # Ensure the covariance matrix is valid (symmetric and positive semi-definite)
            if covariance_z.shape != (d, d):
                raise ValueError(f"Covariance matrix must have shape ({d}, {d})")
            self.covariance_z = covariance_z

    def generate_null(
        self, n_samples: int, noise_scale: float = 1.0, return_params: bool = False
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
            mean=np.zeros(self.d), cov=self.covariance_z, size=n_samples
        )

        # Generate X = Z * beta_x + epsilon_x
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X = Z @ self.beta_x + epsilon_x

        # Generate Y = Z * beta_y + epsilon_y (independent of X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y = Z @ self.beta_y + epsilon_y

        result = {"X": X.reshape(-1, 1), "Y": Y.reshape(-1, 1), "Z": Z}

        if return_params:
            params = {
                "beta_x": self.beta_x,
                "beta_y": self.beta_y,
                "noise_scale": noise_scale,
                "covariance_z": self.covariance_z,
                "hypothesis": "null",
            }
            result["params"] = params

        return result

    def generate_alternative(
        self,
        n_samples: int,
        gamma: float = 0.5,
        noise_scale: float = 1.0,
        return_params: bool = False,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Generate data under the alternative hypothesis (X ⊥̸ Y | Z).

        Args:
            n_samples: Number of samples to generate
            gamma: Strength of dependence between X and Y
            noise_scale: Scale of the noise (standard deviation)
            return_params: Whether to return the parameters used for generation

        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays, and optionally 'params'
        """
        # Generate Z ~ N(0, Σ)
        Z = np.random.multivariate_normal(
            mean=np.zeros(self.d), cov=self.covariance_z, size=n_samples
        )

        # Generate X = Z * beta_x + epsilon_x
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X = Z @ self.beta_x + epsilon_x

        # Generate Y = Z * beta_y + gamma * X + epsilon_y (dependent on X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y = Z @ self.beta_y + gamma * X + epsilon_y

        result = {"X": X.reshape(-1, 1), "Y": Y.reshape(-1, 1), "Z": Z}

        if return_params:
            params = {
                "beta_x": self.beta_x,
                "beta_y": self.beta_y,
                "gamma": gamma,
                "noise_scale": noise_scale,
                "covariance_z": self.covariance_z,
                "hypothesis": "alternative",
            }
            result["params"] = params

        return result

    def generate_dataset(
        self,
        n_samples: int,
        hypothesis: str = "null",
        gamma: float = 0.5,
        noise_scale: float = 1.0,
        return_params: bool = False,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Generate a dataset under either hypothesis.

        Args:
            n_samples: Number of samples to generate
            hypothesis: 'null' or 'alternative'
            gamma: Strength of dependence between X and Y (for alternative)
            noise_scale: Scale of the noise
            return_params: Whether to return the parameters used for generation

        Returns:
            Dictionary containing generated data
        """
        if hypothesis.lower() == "null":
            return self.generate_null(n_samples, noise_scale, return_params)
        elif hypothesis.lower() == "alternative":
            return self.generate_alternative(
                n_samples, gamma, noise_scale, return_params
            )
        else:
            raise ValueError("hypothesis must be 'null' or 'alternative'")
