# Overview

BenchBBoxTest is a Python package designed to provide tools for benchmarking black-box testing methods, with a focus on conditional independence testing in various data domains.

## Package Structure

The package is organized into several main components:

### Core

The core module provides the fundamental abstractions and base classes used throughout the package, including data generators and test interfaces.

### Datasets

The datasets module contains tools for generating synthetic data in different domains:

- **Text**: Language model-based text generation tools
- **Image**: Image generation utilities
- **Simulation**: Data simulation from various distributions

### Evaluation

The evaluation module provides metrics and evaluation tools to assess the performance of testing methods.

### Utils

The utils module includes helper functions and utilities used across the package.

## Key Concepts

### Conditional Independence Testing

A core focus of BenchBBoxTest is conditional independence testing, which examines whether two variables X and Y are independent given a third variable Z.

- **Null Hypothesis (H₀)**: X and Y are conditionally independent given Z (X ⊥ Y | Z)
- **Alternative Hypothesis (H₁)**: X and Y are conditionally dependent given Z (X ⊥̸ Y | Z)

### Data Generators

BenchBBoxTest provides data generators that can create synthetic datasets under both null and alternative hypotheses for benchmarking testing methods. These generators allow you to control:

- Sample size
- Dependency strength
- Noise levels
- Data domain-specific parameters 