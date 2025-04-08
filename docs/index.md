# BenchBBoxTest

A Python package for benchmarking black-box testing methods.

## Overview

BenchBBoxTest provides tools and utilities for generating synthetic datasets and evaluating black-box testing methodologies. The package includes modules for:

- **Text Generation**: Using language models to generate text datasets
- **Image Generation**: Creating image datasets with specific properties
- **Simulation**: Simulating data from various distributions
- **Evaluation**: Tools for evaluating testing methods

## Installation

```bash
pip install benchbboxtest
```

## Quick Start

```python
from benchbboxtest.datasets.text import LLMGenerator

# Initialize a language model generator
generator = LLMGenerator(model_name="gpt2")

# Generate text based on a prompt
text = generator.generate_text(
    prompt="This is an example of text generation",
    max_length=100
)

print(text)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 