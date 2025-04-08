# Contributing to BenchBBoxTest

We welcome contributions to BenchBBoxTest! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up your development environment
4. Create a new branch for your feature or bug fix

## Development Environment

Set up your development environment:

```bash
# Clone the repository
git clone https://github.com/jiantingfeng/BenchBBoxTest.git
cd BenchBBoxTest

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding style of the project

3. Add tests for your changes

4. Run the tests:
   ```bash
   pytest
   ```

5. Update documentation if necessary

## Documentation

We use MkDocs with the Material theme and mkdocstrings for documentation:

1. Install documentation dependencies:
   ```bash
   pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python
   ```

2. Preview documentation locally:
   ```bash
   mkdocs serve
   ```

3. Build documentation:
   ```bash
   mkdocs build
   ```

## Submitting Changes

1. Commit your changes:
   ```bash
   git commit -m "Description of your changes"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Submit a pull request on GitHub

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if necessary
3. The maintainers will review your PR
4. Address any feedback and requested changes
5. Once approved, your PR will be merged

## Code Style

- Follow PEP 8 guidelines
- Use Google-style docstrings
- Use type hints where appropriate

Thank you for contributing to BenchBBoxTest! 