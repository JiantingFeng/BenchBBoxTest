"""
Example script for evaluating conditional independence tests on text data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from benchbboxtest.core import (
    ConditionalRandomizationTest,
    HoldoutRandomizationTest
)
from benchbboxtest.datasets.text import (
    LLMGenerator,
    TextGenerator
)
from benchbboxtest.evaluation import simultaneous_evaluation, plot_evaluation_results
from benchbboxtest.utils.visualization import plot_test_comparison


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create LLM generator (Note: This would load a pre-trained model in a real implementation)
    print("Initializing LLM generator...")
    try:
        llm_gen = LLMGenerator(model_name='gpt2')
        print("LLM generator initialized successfully.")
    except Exception as e:
        print(f"Error initializing LLM generator: {e}")
        print("Using a placeholder implementation instead.")
        llm_gen = None
    
    # Create text data generator
    text_gen = TextGenerator(llm_generator=llm_gen)
    
    # Create conditional independence tests
    crt = ConditionalRandomizationTest(n_permutations=50, random_state=42)
    hrt = HoldoutRandomizationTest(n_permutations=50, test_size=0.3, random_state=42)
    
    # Sample sizes to evaluate
    n_samples_list = [20, 50, 100]
    
    # Evaluate tests on text data
    print("Evaluating tests on text data...")
    results_text = {}
    
    # Note: In a real implementation, this would use actual text data
    # For this example, we're using the placeholder implementation in TextGenerator
    results_text['CRT'] = simultaneous_evaluation(
        test_method=crt,
        data_generator=text_gen,
        n_samples_list=n_samples_list,
        n_trials=5
    )
    
    results_text['HRT'] = simultaneous_evaluation(
        test_method=hrt,
        data_generator=text_gen,
        n_samples_list=n_samples_list,
        n_trials=5
    )
    
    # Plot comparison for text data
    plt_text = plot_test_comparison(
        results_text,
        title="Comparison of CIT Methods on Text Data"
    )
    plt_text.savefig("text_comparison.png")
    
    print("Done! Results saved as PNG files.")


if __name__ == "__main__":
    main() 