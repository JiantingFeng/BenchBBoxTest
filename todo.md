# BenchBBoxTest Implementation Todo List

This implementation plan outlines the step-by-step tasks for developing the BenchBBoxTest package described in the paper, which provides standardized benchmarks for evaluating Conditional Independence Testing (CIT) methods across simulation, image, and text data modalities.

## 1. Project Setup

- Initialize a Python package structure with the name "BenchBBoxTest"
- Set up Git repository and create initial commit
- Configure project structure:

```
BenchBBoxTest/
├── benchbboxtest/
│   ├── __init__.py
│   ├── core/
│   ├── datasets/
│   │   ├── simulation/
│   │   ├── image/
│   │   └── text/
│   ├── evaluation/
│   └── utils/
├── examples/
├── tests/
├── docs/
├── setup.py
├── README.md
└── requirements.txt
```

- Create `setup.py` with package metadata and dependencies
- Write an initial README.md explaining the package purpose and structure


## 2. Core Framework Implementation

- Create abstract base classes in `core/`:
  - `CITest`: Base class for all conditional independence test methods
  - `DataGenerator`: Base interface for dataset generation
  - `Evaluator`: Base class for evaluation metrics
- Implement Algorithm 1 from the paper as `simultaneous_evaluation` in `evaluation/metrics.py`
- Create data structures for storing and analyzing test results


## 3. Simulation-based Datasets

- Implement linear Gaussian models from Shah and Peters (2020):

```python
def generate_linear_gaussian_null(n, d, beta_x, beta_y)
def generate_linear_gaussian_alt(n, d, beta_x, beta_y, gamma)
```

- Implement post-nonlinear models from Doran et al. (2014):

```python
def generate_post_nonlinear_null(n, p, a_x, a_y)
def generate_post_nonlinear_alt(n, p, a_x, a_y, b)
```

- Add parameter control for dependency strength, dimensionality, and sample size
- Create utility functions for dataset visualization and inspection


## 4. Image-based Datasets (Using CelebAMask-HQ)

- Implement data downloading and processing utilities for CelebAMask-HQ
- Create segmentation mask handling functions for region selection
- Implement masking operators for facial regions:

```python
def mask_region(image, segmentation_mask, region_name)
```

- Build null hypothesis dataset generators (masking non-crucial/symmetric regions)
- Build alternative hypothesis dataset generators (masking crucial regions)
- Create attribute-region mapping dictionary (e.g., "Narrow_Eyes" → "right_eye")
- Add validation utilities to verify conditional independence relationships


## 5. Text-based Datasets (Using LLMs and arXiv papers)

- Implement an arXiv paper collector for papers published after LLM cutoff date
- Set up LLM integration (using GPT-2 or similar):

```python
class LLMGenerator:
    def generate_text(self, prompt, max_length=100)
```

- Create null hypothesis text generators (context-only prompts)
- Create alternative hypothesis text generators (context+label prompts)
- Implement dataset preprocessing and vectorization utilities
- Add validation methods to verify conditional independence in text data


## 6. Evaluation Framework

- Implement baseline CIT methods:
  - Conditional Randomization Test (CRT)
  - Conditional Permutation Test (CPT)
  - Holdout Randomization Test (HRT)
  - Projected Covariance Measure
- Create standardized evaluation protocols for each data modality
- Implement visualization tools for comparing Type I error and power
- Add utilities for statistical significance analysis of test results