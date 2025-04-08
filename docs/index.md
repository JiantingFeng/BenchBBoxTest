# BenchBBoxTest

A Python package for benchmarking black-box conditional independence testing methods across multiple data modalities.

## Overview

Conditional Independence Testing (CIT) is a fundamental task in causal inference and statistical analysis. BenchBBoxTest provides standardized benchmarks and data generators for evaluating CIT methods across different types of data:

1.  **Simulation data**: Linear Gaussian and Post-nonlinear models.
2.  **Image data**: Using the CelebAMask-HQ dataset with facial attributes. See [Image Data Generation](datasets/image.md).
3.  **Text data**: Using Large Language Models (LLMs) and scenarios like synthetic EHR notes. See [Text Data Generation](datasets/text.md).

The package includes tools for data generation, CIT method implementations (like CRT), and evaluation utilities.

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

Alternatively, install the latest release via pip:
```bash
pip install benchbboxtest
```

## Quick Start

Here's a basic example demonstrating CIT evaluation:

```python
import numpy as np
from benchbboxtest.core import ConditionalRandomizationTest
from benchbboxtest.datasets.simulation import LinearGaussianGenerator
from benchbboxtest.evaluation import simultaneous_evaluation, plot_evaluation_results

# Create a data generator (simulation data)
data_gen = LinearGaussianGenerator(d=5)

# Create a conditional independence test method
cit = ConditionalRandomizationTest(n_permutations=100)

# Evaluate the test
results = simultaneous_evaluation(
    test_method=cit,
    data_generator=data_gen,
    n_samples_list=[100, 200, 500, 1000],
    n_trials=10
)

# Plot the results (e.g., power curve)
plot_evaluation_results(results)
```

For more examples, including text and image data, see the [Getting Started](getting-started.md) guide.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 