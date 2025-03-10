"""
Example script for evaluating conditional independence tests on image data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from benchbboxtest.core import (
    ConditionalRandomizationTest,
    HoldoutRandomizationTest
)
from benchbboxtest.datasets.image import (
    CelebAMaskGenerator,
    download_celebamask_hq
)
from benchbboxtest.evaluation import simultaneous_evaluation, plot_evaluation_results
from benchbboxtest.utils.visualization import plot_test_comparison


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set up dataset directory
    dataset_dir = os.path.join(os.getcwd(), "data", "celebamask-hq")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Check if dataset exists, if not, download it (placeholder)
    if not os.path.exists(os.path.join(dataset_dir, "CelebA-HQ-img")):
        print("Dataset not found. Downloading CelebAMask-HQ dataset...")
        download_celebamask_hq(dataset_dir)
        print("Note: This is a placeholder. In a real implementation, you would need to download the dataset manually.")
    
    # Create data generators for different attributes
    print("Creating data generators...")
    smiling_gen = CelebAMaskGenerator(dataset_path=dataset_dir, attribute="Smiling")
    eyeglasses_gen = CelebAMaskGenerator(dataset_path=dataset_dir, attribute="Eyeglasses")
    
    # Create conditional independence tests
    crt = ConditionalRandomizationTest(n_permutations=50, random_state=42)
    hrt = HoldoutRandomizationTest(n_permutations=50, test_size=0.3, random_state=42)
    
    # Sample sizes to evaluate
    n_samples_list = [50, 100, 200]
    
    # Evaluate tests on "Smiling" attribute
    print("Evaluating tests on 'Smiling' attribute...")
    results_smiling = {}
    
    # Note: In a real implementation, this would use actual image data
    # For this example, we're using the placeholder implementation in CelebAMaskGenerator
    results_smiling['CRT'] = simultaneous_evaluation(
        test_method=crt,
        data_generator=smiling_gen,
        n_samples_list=n_samples_list,
        n_trials=5
    )
    
    results_smiling['HRT'] = simultaneous_evaluation(
        test_method=hrt,
        data_generator=smiling_gen,
        n_samples_list=n_samples_list,
        n_trials=5
    )
    
    # Plot comparison for "Smiling" attribute
    plt_smiling = plot_test_comparison(
        results_smiling,
        title="Comparison of CIT Methods on 'Smiling' Attribute"
    )
    plt_smiling.savefig("smiling_comparison.png")
    
    # Evaluate tests on "Eyeglasses" attribute
    print("Evaluating tests on 'Eyeglasses' attribute...")
    results_eyeglasses = {}
    
    results_eyeglasses['CRT'] = simultaneous_evaluation(
        test_method=crt,
        data_generator=eyeglasses_gen,
        n_samples_list=n_samples_list,
        n_trials=5
    )
    
    results_eyeglasses['HRT'] = simultaneous_evaluation(
        test_method=hrt,
        data_generator=eyeglasses_gen,
        n_samples_list=n_samples_list,
        n_trials=5
    )
    
    # Plot comparison for "Eyeglasses" attribute
    plt_eyeglasses = plot_test_comparison(
        results_eyeglasses,
        title="Comparison of CIT Methods on 'Eyeglasses' Attribute"
    )
    plt_eyeglasses.savefig("eyeglasses_comparison.png")
    
    print("Done! Results saved as PNG files.")


if __name__ == "__main__":
    main() 