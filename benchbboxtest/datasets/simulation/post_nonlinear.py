import numpy as np
from typing import Dict, Tuple, Optional, Callable
from ...core import DataGenerator


class PostNonlinearGenerator(DataGenerator):
    """
    Post-nonlinear models from Doran et al. (2014).
    
    Implements data generation for post-nonlinear models under both
    null and alternative hypotheses.
    """
    
    def __init__(self, p: int = 5, a_x: Optional[np.ndarray] = None, 
                 a_y: Optional[np.ndarray] = None,
                 nonlinear_func: Callable = np.tanh):
        """
        Initialize the post-nonlinear generator.
        
        Args:
            p: Dimension of the conditioning variable Z
            a_x: Coefficients for X ~ Z relationship (if None, random values are generated)
            a_y: Coefficients for Y ~ Z relationship (if None, random values are generated)
            nonlinear_func: Nonlinear function to apply (default: tanh)
        """
        self.p = p
        self.nonlinear_func = nonlinear_func
        
        # Initialize coefficients if not provided
        if a_x is None:
            self.a_x = np.random.normal(0, 1, size=p)
        else:
            self.a_x = a_x
            
        if a_y is None:
            self.a_y = np.random.normal(0, 1, size=p)
        else:
            self.a_y = a_y
    
    def generate_null(self, n_samples: int, noise_scale: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Generate data under the null hypothesis (X ⊥ Y | Z).
        
        Args:
            n_samples: Number of samples to generate
            noise_scale: Scale of the noise
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # Generate Z ~ N(0, I)
        Z = np.random.normal(0, 1, size=(n_samples, self.p))
        
        # Generate X = f(Z * a_x + epsilon_x)
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X_linear = Z @ self.a_x + epsilon_x
        X = self.nonlinear_func(X_linear)
        
        # Generate Y = f(Z * a_y + epsilon_y) (independent of X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y_linear = Z @ self.a_y + epsilon_y
        Y = self.nonlinear_func(Y_linear)
        
        return {'X': X.reshape(-1, 1), 'Y': Y.reshape(-1, 1), 'Z': Z}
    
    def generate_alternative(self, n_samples: int, b: float = 0.5, 
                            noise_scale: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Generate data under the alternative hypothesis (X ⊥̸ Y | Z).
        
        Args:
            n_samples: Number of samples to generate
            b: Strength of dependence between X and Y
            noise_scale: Scale of the noise
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # Generate Z ~ N(0, I)
        Z = np.random.normal(0, 1, size=(n_samples, self.p))
        
        # Generate X = f(Z * a_x + epsilon_x)
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X_linear = Z @ self.a_x + epsilon_x
        X = self.nonlinear_func(X_linear)
        
        # Generate Y = f(Z * a_y + b * X + epsilon_y) (dependent on X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y_linear = Z @ self.a_y + b * X + epsilon_y
        Y = self.nonlinear_func(Y_linear)
        
        return {'X': X.reshape(-1, 1), 'Y': Y.reshape(-1, 1), 'Z': Z} 