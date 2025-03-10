import numpy as np
from typing import Dict, List, Tuple, Callable, Any
import matplotlib.pyplot as plt
from ..core import CITest, DataGenerator


def simultaneous_evaluation(
    test_method: CITest,
    data_generator: DataGenerator,
    n_samples_list: List[int],
    n_trials: int = 100,
    alpha: float = 0.05,
    **generator_kwargs
) -> Dict[str, Dict[str, List[float]]]:
    """
    Implementation of Algorithm 1 from the paper: Simultaneous evaluation of Type I error and power.
    
    Args:
        test_method: The conditional independence test to evaluate
        data_generator: The data generator to use
        n_samples_list: List of sample sizes to evaluate
        n_trials: Number of trials for each sample size
        alpha: Significance level
        **generator_kwargs: Additional parameters for the data generator
        
    Returns:
        Dictionary containing evaluation results:
            - 'null': Dictionary with 'rejection_rates' for each sample size
            - 'alternative': Dictionary with 'rejection_rates' for each sample size
    """
    results = {
        'null': {'rejection_rates': [], 'sample_sizes': n_samples_list},
        'alternative': {'rejection_rates': [], 'sample_sizes': n_samples_list}
    }
    
    for n_samples in n_samples_list:
        # Type I error evaluation (null hypothesis)
        null_rejections = 0
        for _ in range(n_trials):
            data = data_generator.generate_null(n_samples, **generator_kwargs)
            reject, _ = test_method.test(data['X'], data['Y'], data['Z'], alpha=alpha)
            if reject:
                null_rejections += 1
        
        type_I_error = null_rejections / n_trials
        results['null']['rejection_rates'].append(type_I_error)
        
        # Power evaluation (alternative hypothesis)
        alt_rejections = 0
        for _ in range(n_trials):
            data = data_generator.generate_alternative(n_samples, **generator_kwargs)
            reject, _ = test_method.test(data['X'], data['Y'], data['Z'], alpha=alpha)
            if reject:
                alt_rejections += 1
        
        power = alt_rejections / n_trials
        results['alternative']['rejection_rates'].append(power)
    
    return results


def plot_evaluation_results(results: Dict[str, Dict[str, List[float]]], 
                           title: str = "Conditional Independence Test Evaluation"):
    """
    Plot the evaluation results from simultaneous_evaluation.
    
    Args:
        results: Results dictionary from simultaneous_evaluation
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    sample_sizes = results['null']['sample_sizes']
    null_rates = results['null']['rejection_rates']
    alt_rates = results['alternative']['rejection_rates']
    
    plt.plot(sample_sizes, null_rates, 'o-', label='Type I Error (Null)')
    plt.plot(sample_sizes, alt_rates, 's-', label='Power (Alternative)')
    plt.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Rejection Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt 