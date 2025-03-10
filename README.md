# BenchBBoxTest

BenchBBoxTest is a Python package for benchmarking black-box conditional independence testing methods across multiple data modalities.

## Overview

Conditional Independence Testing (CIT) is a fundamental task in causal inference and statistical analysis. This package provides standardized benchmarks for evaluating CIT methods across:

1. **Simulation data**: Linear Gaussian and Post-nonlinear models
2. **Image data**: Using CelebAMask-HQ dataset with facial attributes
3. **Text data**: Using LLMs and arXiv papers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BenchBBoxTest.git
cd BenchBBoxTest

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from benchbboxtest.core import ConditionalRandomizationTest
from benchbboxtest.datasets.simulation import LinearGaussianGenerator
from benchbboxtest.evaluation import simultaneous_evaluation, plot_evaluation_results

# Create a data generator
data_gen = LinearGaussianGenerator(d=5)

# Create a conditional independence test
cit = ConditionalRandomizationTest(n_permutations=100)

# Evaluate the test
results = simultaneous_evaluation(
    test_method=cit,
    data_generator=data_gen,
    n_samples_list=[100, 200, 500, 1000],
    n_trials=10
)

# Plot the results
plot_evaluation_results(results)
```

### Available Data Generators

#### Simulation Data
- `LinearGaussianGenerator`: Linear Gaussian models
- `PostNonlinearGenerator`: Post-nonlinear models

#### Image Data
- `CelebAMaskGenerator`: CelebAMask-HQ dataset with facial attributes

#### Text Data
- `TextGenerator`: Text data using LLMs and arXiv papers

### Available CIT Methods

- `ConditionalRandomizationTest`: CRT method
- `HoldoutRandomizationTest`: HRT method
- `ProjectedCovarianceTest`: Projected covariance measure

## Project Structure

```
BenchBBoxTest/
├── benchbboxtest/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── methods.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── simulation/
│   │   ├── image/
│   │   └── text/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
├── examples/
├── tests/
├── docs/
├── setup.py
├── README.md
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

