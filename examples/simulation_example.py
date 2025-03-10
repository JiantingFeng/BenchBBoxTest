"""
Example script for evaluating conditional independence tests on simulation data.
"""

import numpy as np
import matplotlib.pyplot as plt
from benchbboxtest.core import (
    ConditionalRandomizationTest,
    HoldoutRandomizationTest,
    ProjectedCovarianceTest
)
from benchbboxtest.datasets.simulation import (
    LinearGaussianGenerator,
    PostNonlinearGenerator
)
from benchbboxtest.evaluation import simultaneous_evaluation, plot_evaluation_results
from benchbboxtest.utils.visualization import plot_test_comparison


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create data generators
    linear_gen = LinearGaussianGenerator(d=5)
    nonlinear_gen = PostNonlinearGenerator(p=5)
    
    # Create conditional independence tests
    crt = ConditionalRandomizationTest(n_permutations=100, random_state=42)
    hrt = HoldoutRandomizationTest(n_permutations=100, test_size=0.3, random_state=42)
    pct = ProjectedCovarianceTest(n_permutations=100, random_state=42)
    
    # Sample sizes to evaluate
    n_samples_list = [100, 200, 500, 1000]
    
    # Evaluate tests on linear Gaussian data
    print("Evaluating tests on linear Gaussian data...")
    results_linear = {}
    
    results_linear['CRT'] = simultaneous_evaluation(
        test_method=crt,
        data_generator=linear_gen,
        n_samples_list=n_samples_list,
        n_trials=10,
        gamma=0.5  # Strength of dependence for alternative hypothesis
    )
    
    results_linear['HRT'] = simultaneous_evaluation(
        test_method=hrt,
        data_generator=linear_gen,
        n_samples_list=n_samples_list,
        n_trials=10,
        gamma=0.5
    )
    
    results_linear['PCT'] = simultaneous_evaluation(
        test_method=pct,
        data_generator=linear_gen,
        n_samples_list=n_samples_list,
        n_trials=10,
        gamma=0.5
    )
    
    # Plot comparison for linear Gaussian data
    plt_linear = plot_test_comparison(
        results_linear,
        title="Comparison of CIT Methods on Linear Gaussian Data"
    )
    plt_linear.savefig("linear_gaussian_comparison.png")
    
    # Evaluate tests on post-nonlinear data
    print("Evaluating tests on post-nonlinear data...")
    results_nonlinear = {}
    
    results_nonlinear['CRT'] = simultaneous_evaluation(
        test_method=crt,
        data_generator=nonlinear_gen,
        n_samples_list=n_samples_list,
        n_trials=10,
        b=0.5  # Strength of dependence for alternative hypothesis
    )
    
    results_nonlinear['HRT'] = simultaneous_evaluation(
        test_method=hrt,
        data_generator=nonlinear_gen,
        n_samples_list=n_samples_list,
        n_trials=10,
        b=0.5
    )
    
    results_nonlinear['PCT'] = simultaneous_evaluation(
        test_method=pct,
        data_generator=nonlinear_gen,
        n_samples_list=n_samples_list,
        n_trials=10,
        b=0.5
    )
    
    # Plot comparison for post-nonlinear data
    plt_nonlinear = plot_test_comparison(
        results_nonlinear,
        title="Comparison of CIT Methods on Post-Nonlinear Data"
    )
    plt_nonlinear.savefig("post_nonlinear_comparison.png")
    
    print("Done! Results saved as PNG files.")


if __name__ == "__main__":
    main() 