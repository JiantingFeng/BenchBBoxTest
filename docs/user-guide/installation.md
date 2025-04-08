# Installation

This page provides detailed instructions for installing BenchBBoxTest in different environments.

## Requirements

BenchBBoxTest requires:

- Python 3.8 or higher
- PyTorch (for language model functionality)
- NumPy
- Transformers (for pre-trained language models)

## Standard Installation

The easiest way to install BenchBBoxTest is using pip:

```bash
pip install benchbboxtest
```

This will install the package and all its required dependencies.

## Installation from Source

For the latest features or to contribute to development, you can install from source:

```bash
git clone https://github.com/jiantingfeng/BenchBBoxTest.git
cd BenchBBoxTest
pip install -e .
```

## GPU Support

Some features of BenchBBoxTest, particularly the language model functionality, can benefit significantly from GPU acceleration. To enable GPU support:

1. Make sure you have the appropriate CUDA drivers installed
2. Install PyTorch with CUDA support following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/)

## Optional Dependencies

### For OpenAI Integration

To use the OpenAI API functionality:

```bash
pip install openai
```

### For ArXiv Integration

To use the ArXiv paper collection functionality:

```bash
pip install arxiv
``` 