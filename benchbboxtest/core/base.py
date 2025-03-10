from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union


class CITest(ABC):
    """
    Abstract base class for all conditional independence test methods.
    
    All conditional independence tests should implement this interface.
    The test evaluates whether X ⊥ Y | Z (X is independent of Y given Z).
    """
    
    @abstractmethod
    def test(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
             alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Perform a conditional independence test.
        
        Args:
            X: The first variable
            Y: The second variable
            Z: The conditioning variable
            alpha: Significance level
            
        Returns:
            Tuple containing:
                - Boolean indicating rejection of the null hypothesis (True if X and Y are dependent given Z)
                - p-value or test statistic
        """
        pass


class DataGenerator(ABC):
    """
    Abstract base class for dataset generation.
    
    All data generators should implement this interface to provide
    consistent dataset generation across different modalities.
    """
    
    @abstractmethod
    def generate_null(self, n_samples: int, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate data under the null hypothesis (X ⊥ Y | Z).
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters specific to the generator
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        pass
    
    @abstractmethod
    def generate_alternative(self, n_samples: int, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate data under the alternative hypothesis (X ⊥̸ Y | Z).
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters specific to the generator
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        pass


class Evaluator(ABC):
    """
    Abstract base class for evaluation metrics.
    
    This class defines the interface for evaluating conditional independence tests.
    """
    
    @abstractmethod
    def evaluate(self, test_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the performance of a conditional independence test.
        
        Args:
            test_results: Dictionary containing test results
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass 