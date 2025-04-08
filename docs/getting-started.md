# Getting Started

This guide will help you get started with BenchBBoxTest.

## Installation

You can install BenchBBoxTest using pip:

```bash
pip install benchbboxtest
```

Or you can install from source:

```bash
git clone https://github.com/yourusername/BenchBBoxTest.git
cd BenchBBoxTest
pip install -e .
```

## Basic Usage

### Text Generation

BenchBBoxTest provides several text generation tools powered by language models:

```python
from benchbboxtest.datasets.text import LLMGenerator, TextGenerator

# Initialize a language model generator
llm = LLMGenerator(model_name="gpt2")

# Generate text based on a prompt
texts = llm.generate_text(
    prompt="Write a paragraph about machine learning.",
    max_length=150,
    temperature=0.7
)

# Create a text generator for conditional independence testing
generator = TextGenerator(llm_generator=llm)

# Generate data under the null hypothesis
null_data = generator.generate_null(
    n_samples=100,
    temperature=0.8
)

# Generate data under the alternative hypothesis
alt_data = generator.generate_alternative(
    n_samples=100,
    dependency_strength=0.7
)
```

### Using OpenAI API

For more powerful text generation, you can use the OpenAI API integration:

```python
from benchbboxtest.datasets.text import OpenAIGenerator

# Initialize the OpenAI generator with your API key
# It will use the OPENAI_API_KEY environment variable if not provided
generator = OpenAIGenerator(
    api_key="your-api-key",  # or set OPENAI_API_KEY environment variable
    model="gpt-3.5-turbo"
)

# Generate text
texts = generator.generate_text(
    prompt="Explain the concept of conditional independence.",
    max_tokens=150
)
```

## Next Steps

Refer to the User Guide and API Reference for more detailed information on the various modules and functionality provided by BenchBBoxTest. 