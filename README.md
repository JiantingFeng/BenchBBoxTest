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
git clone https://github.com/jiantingfeng/BenchBBoxTest.git
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

### EHR Text Data Example

This example demonstrates how to use the `EHRTextGenerator` to create synthetic clinical notes for CIT benchmarks. The generator simulates scenarios where clinical notes (X) might be conditionally independent of a diagnosis (Y) given patient background (Z) (null hypothesis) or dependent (alternative hypothesis).

```python
from benchbboxtest.datasets.text import EHRTextGenerator
from benchbboxtest.core import ConditionalRandomizationTest
from benchbboxtest.evaluation import evaluate_cit
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Setup the EHR Text Generator ---
# Optional: Define a text vectorizer (e.g., TF-IDF)
def text_vectorizer(texts):
    vectorizer = TfidfVectorizer(max_features=100) # Limit features for example
    return vectorizer.fit_transform(texts).toarray()

# Initialize the generator (requires a language model, defaults to GPT-2)
# For OpenAI models, use:
# from benchbboxtest.datasets.text import OpenAIGenerator
# llm = OpenAIGenerator(api_key='YOUR_API_KEY', model='gpt-3.5-turbo')
# ehr_gen = EHRTextGenerator(llm_generator=llm, vectorizer=text_vectorizer)
ehr_gen = EHRTextGenerator(vectorizer=text_vectorizer)

# --- 2. Generate Data ---
# Generate data under the alternative hypothesis (notes depend on diagnosis)
data_alt = ehr_gen.generate_alternative(n_samples=50, dependency_strength=0.9)
X_alt, Y_alt, Z_alt = data_alt['X'], data_alt['Y'], data_alt['Z'] # Z contains vectorized patient info

# Generate data under the null hypothesis (notes independent of diagnosis given patient info)
data_null = ehr_gen.generate_null(n_samples=50)
X_null, Y_null, Z_null = data_null['X'], data_null['Y'], data_null['Z']

# --- 3. Evaluate CIT Test ---
# Initialize a CIT test method
cit_test = ConditionalRandomizationTest(n_permutations=200)

# Evaluate on alternative data (expect low p-value, reject null)
p_value_alt, _ = evaluate_cit(cit_test, X_alt, Y_alt, Z_alt)
print(f"P-value (Alternative Hypothesis): {p_value_alt:.4f}")

# Evaluate on null data (expect high p-value, fail to reject null)
p_value_null, _ = evaluate_cit(cit_test, X_null, Y_null, Z_null)
print(f"P-value (Null Hypothesis): {p_value_null:.4f}")

# You can also generate a human-readable sample:
# example_data = ehr_gen.generate_example_dataset(n_samples=2, hypothesis='alternative')
# print(example_data)
```

### Available Data Generators

#### Simulation Data
- `LinearGaussianGenerator`: Generates data from linear Gaussian models. Located in `benchbboxtest/datasets/simulation/linear_gaussian.py`.
- `PostNonlinearGenerator`: Generates data from post-nonlinear models. Located in `benchbboxtest/datasets/simulation/post_nonlinear.py`.

#### Image Data
- `CelebAMaskGenerator`: Generates data using the CelebAMask-HQ dataset with facial attributes. Located in `benchbboxtest/datasets/image/celebamask.py`.

#### Text Data
- `TextGenerator`: Base class for text data generation. Located in `benchbboxtest/datasets/text/llm_text.py`.
- `LLMGenerator`: Generates text data using Large Language Models (LLMs). Located in `benchbboxtest/datasets/text/llm_text.py`.
- `ArXivCollector`: Collects text data from arXiv papers. Located in `benchbboxtest/datasets/text/llm_text.py`.

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
│   │   │   ├── __init__.py
│   │   │   ├── linear_gaussian.py
│   │   │   └── post_nonlinear.py
│   │   ├── image/
│   │   │   ├── __init__.py
│   │   │   └── celebamask.py
│   │   └── text/
│   │       ├── __init__.py
│   │       └── llm_text.py
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

