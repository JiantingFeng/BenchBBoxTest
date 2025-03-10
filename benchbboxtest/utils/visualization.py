import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns


def plot_data_distribution(X: np.ndarray, Y: np.ndarray, Z: np.ndarray = None, 
                          title: str = "Data Distribution"):
    """
    Plot the distribution of the data.
    
    Args:
        X: The first variable
        Y: The second variable
        Z: The conditioning variable (optional)
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    if Z is not None and Z.shape[1] <= 2:
        # If Z is 1D or 2D, we can visualize it
        if Z.shape[1] == 1:
            # 3D plot with X, Y, Z
            ax = plt.subplot(111, projection='3d')
            ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=Y.ravel(), cmap='viridis', alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            # 3D plot with X, Y, Z[0], colored by Z[1]
            ax = plt.subplot(111, projection='3d')
            ax.scatter(X.ravel(), Y.ravel(), Z[:, 0], c=Z[:, 1], cmap='viridis', alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z[0]')
            plt.colorbar(ax.scatter(X.ravel(), Y.ravel(), Z[:, 0], c=Z[:, 1], cmap='viridis', alpha=0.6), 
                         label='Z[1]')
    else:
        # Simple 2D scatter plot of X vs Y
        plt.scatter(X.ravel(), Y.ravel(), alpha=0.6)
        plt.xlabel('X')
        plt.ylabel('Y')
    
    plt.title(title)
    plt.tight_layout()
    
    return plt


def plot_test_comparison(results_dict: Dict[str, Dict[str, Dict[str, List[float]]]],
                        title: str = "Comparison of Conditional Independence Tests"):
    """
    Plot a comparison of multiple conditional independence tests.
    
    Args:
        results_dict: Dictionary mapping test names to results dictionaries from simultaneous_evaluation
        title: Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots for Type I error and power
    plt.subplot(1, 2, 1)
    for test_name, results in results_dict.items():
        sample_sizes = results['null']['sample_sizes']
        null_rates = results['null']['rejection_rates']
        plt.plot(sample_sizes, null_rates, 'o-', label=f'{test_name}')
    
    plt.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
    plt.xlabel('Sample Size')
    plt.ylabel('Type I Error Rate')
    plt.title('Type I Error Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for test_name, results in results_dict.items():
        sample_sizes = results['alternative']['sample_sizes']
        alt_rates = results['alternative']['rejection_rates']
        plt.plot(sample_sizes, alt_rates, 's-', label=f'{test_name}')
    
    plt.xlabel('Sample Size')
    plt.ylabel('Power')
    plt.title('Power Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return plt


def plot_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str],
                title: str = "Heatmap", cmap: str = "viridis"):
    """
    Plot a heatmap of a matrix.
    
    Args:
        matrix: Matrix to plot
        row_labels: Labels for the rows
        col_labels: Labels for the columns
        title: Plot title
        cmap: Colormap to use
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(matrix, annot=True, cmap=cmap, xticklabels=col_labels, yticklabels=row_labels)
    
    plt.title(title)
    plt.tight_layout()
    
    return plt 