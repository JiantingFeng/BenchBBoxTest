import numpy as np
from typing import Dict, Tuple, Optional
from ...core import DataGenerator


class LinearGaussianGenerator(DataGenerator):
    """
    Linear Gaussian models from Shah and Peters (2020).
    
    Implements data generation for linear Gaussian models under both
    null and alternative hypotheses.
    """
    
    def __init__(self, d: int = 5, beta_x: Optional[np.ndarray] = None, 
                 beta_y: Optional[np.ndarray] = None):
        """
        Initialize the linear Gaussian generator.
        
        Args:
            d: Dimension of the conditioning variable Z
            beta_x: Coefficients for X ~ Z relationship (if None, random values are generated)
            beta_y: Coefficients for Y ~ Z relationship (if None, random values are generated)
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
    
    def generate_null(self, n_samples: int, noise_scale: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate data under the null hypothesis (X ⊥ Y | Z).
        
        Args:
            n_samples: Number of samples to generate
            noise_scale: Scale of the noise
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # Generate Z ~ N(0, I)
        Z = np.random.normal(0, 1, size=(n_samples, self.d))
        
        # Generate X = Z * beta_x + epsilon_x
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X = Z @ self.beta_x + epsilon_x
        
        # Generate Y = Z * beta_y + epsilon_y (independent of X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y = Z @ self.beta_y + epsilon_y
        
        return {'X': X.reshape(-1, 1), 'Y': Y.reshape(-1, 1), 'Z': Z}
    
    def generate_alternative(self, n_samples: int, gamma: float = 0.5, 
                            noise_scale: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate data under the alternative hypothesis (X ⊥̸ Y | Z).
        
        Args:
            n_samples: Number of samples to generate
            gamma: Strength of dependence between X and Y
            noise_scale: Scale of the noise
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # Generate Z ~ N(0, I)
        Z = np.random.normal(0, 1, size=(n_samples, self.d))
        
        # Generate X = Z * beta_x + epsilon_x
        epsilon_x = np.random.normal(0, noise_scale, size=n_samples)
        X = Z @ self.beta_x + epsilon_x
        
        # Generate Y = Z * beta_y + gamma * X + epsilon_y (dependent on X given Z)
        epsilon_y = np.random.normal(0, noise_scale, size=n_samples)
        Y = Z @ self.beta_y + gamma * X + epsilon_y
        
        return {'X': X.reshape(-1, 1), 'Y': Y.reshape(-1, 1), 'Z': Z} 