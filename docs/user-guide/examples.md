# Examples

This page provides practical examples of using BenchBBoxTest for various tasks.

## Text Generation Examples

### Basic Text Generation with LLMGenerator

```python
from benchbboxtest.datasets.text import LLMGenerator

# Initialize the generator
generator = LLMGenerator(model_name="gpt2")

# Generate text with different parameters
text1 = generator.generate_text(
    prompt="Artificial intelligence is",
    max_length=100,
    temperature=0.7
)[0]

text2 = generator.generate_text(
    prompt="Artificial intelligence is",
    max_length=100,
    temperature=1.5,  # Higher temperature = more random
    num_return_sequences=3  # Return multiple sequences
)

print(f"Generated text (temperature=0.7):\n{text1}\n")
print(f"Generated texts (temperature=1.5):")
for i, text in enumerate(text2):
    print(f"Sequence {i+1}:\n{text}\n")
```

### Using OpenAI API

```python
import os
from benchbboxtest.datasets.text import OpenAIGenerator

# Set your API key as an environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize the generator
generator = OpenAIGenerator(model="gpt-3.5-turbo")

# Generate text using the chat model
response = generator.generate_text(
    prompt="Explain the concept of black-box testing in simple terms.",
    max_tokens=150,
    temperature=0.7,
    system_message="You are a helpful assistant that explains complex topics simply."
)[0]

print(response)
```

## Conditional Independence Testing

### Generating Null and Alternative Datasets

```python
from benchbboxtest.datasets.text import LLMGenerator, TextGenerator

# Initialize generators
llm = LLMGenerator(model_name="gpt2")
text_gen = TextGenerator(llm_generator=llm)

# Generate data under the null hypothesis (X ⊥ Y | Z)
null_data = text_gen.generate_null(
    n_samples=50,
    temperature=0.8,
    max_length=150
)

# Generate data under the alternative hypothesis (X ⊥̸ Y | Z)
# with strong dependency between X and Y given Z
alt_data_strong = text_gen.generate_alternative(
    n_samples=50,
    dependency_strength=0.9,  # Strong dependency
    temperature=0.8,
    max_length=150
)

# Generate data with weaker dependency
alt_data_weak = text_gen.generate_alternative(
    n_samples=50,
    dependency_strength=0.3,  # Weak dependency
    temperature=0.8,
    max_length=150
)

# Check the shapes of the generated data
print(f"Null data shapes:")
print(f"X: {null_data['X'].shape}, Y: {null_data['Y'].shape}, Z: {null_data['Z'].shape}")
```

### Using EHR Text Generator

```python
from benchbboxtest.datasets.text import LLMGenerator, EHRTextGenerator

# Initialize generators
llm = LLMGenerator(model_name="gpt2")
ehr_gen = EHRTextGenerator(llm_generator=llm)

# Generate a small example dataset under the alternative hypothesis
examples = ehr_gen.generate_example_dataset(
    n_samples=3,
    hypothesis="alternative"
)

# Print the results
for i in range(len(examples["clinical_notes"])):
    print(f"EXAMPLE {i+1}")
    print(f"Patient Info: {examples['patient_info'][i]}")
    print(f"Diagnosis: {'Positive' if examples['diagnoses'][i] == 1 else 'Negative'}")
    print(f"Clinical Note: {examples['clinical_notes'][i][:200]}...\n") 