# Getting Started

This guide will help you get started with BenchBBoxTest.

For a quick overview and basic usage examples, please also refer to the main [README.md](../README.md) file.

## Installation

```bash
# Clone the repository
git clone https://github.com/jiantingfeng/BenchBBoxTest.git
cd BenchBBoxTest

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

Alternatively, you can install the latest release directly using pip:

```bash
pip install benchbboxtest
```

## Basic Usage

Here's a basic example demonstrating how to use BenchBBoxTest for conditional independence testing:

```python
import numpy as np
from benchbboxtest.core import ConditionalRandomizationTest
from benchbboxtest.datasets.simulation import LinearGaussianGenerator
from benchbboxtest.evaluation import simultaneous_evaluation, plot_evaluation_results

# Create a data generator
data_gen = LinearGaussianGenerator(d=5)

# Create a conditional independence test
cit = ConditionalRandomizationTest(n_permutations=100)

# Evaluate the test across different sample sizes
results = simultaneous_evaluation(
    test_method=cit,
    data_generator=data_gen,
    n_samples_list=[100, 200, 500, 1000],
    n_trials=10
)

# Plot the evaluation results (e.g., power curve)
plot_evaluation_results(results)
```

## Examples with Different Data Types

BenchBBoxTest also supports benchmarking with other data modalities:

*   **Image Data:** Use `CelebAMaskGenerator` (`benchbboxtest/datasets/image/celebamask.py`) for CIT tasks involving facial attributes from the CelebAMask-HQ dataset.
*   **Text Data:** Use generators like `LLMGenerator` or `EHRTextGenerator` (`benchbboxtest/datasets/text/`) to create text-based scenarios. See the main `README.md` for a detailed example using `EHRTextGenerator`.

## Next Steps

Refer to the User Guide and API Reference for more detailed information on the various modules and functionality provided by BenchBBoxTest. 