# Text Data Generation

BenchBBoxTest provides several tools for generating text-based datasets suitable for conditional independence testing benchmarks, primarily using Large Language Models (LLMs).

## LLM Generators

Located in `benchbboxtest/datasets/text/llm_text.py`.

-   `LLMGenerator`: A base class leveraging models available through the Hugging Face `transformers` library (e.g., "gpt2").
-   `OpenAIGenerator`: Integrates with the OpenAI API for access to models like "gpt-3.5-turbo" or "gpt-4". Requires an API key (set via `api_key` argument or `OPENAI_API_KEY` environment variable).

```python
# Example using a local model
from benchbboxtest.datasets.text import LLMGenerator
llm_local = LLMGenerator(model_name="gpt2")
texts = llm_local.generate_text(prompt="Example prompt", max_length=50)

# Example using OpenAI
from benchbboxtest.datasets.text import OpenAIGenerator
# generator = OpenAIGenerator(api_key="YOUR_API_KEY", model="gpt-3.5-turbo")
# texts = generator.generate_text(prompt="Another example", max_tokens=50)
```

## EHR Text Generator Example

`EHRTextGenerator` (also in `benchbboxtest/datasets/text/`) provides a specific scenario for CIT benchmarks using synthetic clinical notes.

### Purpose

This generator simulates scenarios where:
- **X**: Clinical notes (text).
- **Y**: A diagnosis or outcome (e.g., binary indicator).
- **Z**: Patient background information (e.g., demographics, lab values, represented numerically or categorically).

The goal is to test if the clinical notes (X) provide additional information about the diagnosis (Y) beyond what's already contained in the patient background (Z).
- **Null Hypothesis**: Notes are independent of diagnosis given patient background (X ⊥ Y | Z).
- **Alternative Hypothesis**: Notes are dependent on diagnosis given patient background.

### Usage Example

This example demonstrates generating data under both hypotheses and evaluating a CIT test.

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

# Initialize the generator (defaults to GPT-2)
# To use OpenAI, provide an OpenAIGenerator instance:
# from benchbboxtest.datasets.text import OpenAIGenerator
# llm = OpenAIGenerator(api_key='YOUR_API_KEY', model='gpt-3.5-turbo')
# ehr_gen = EHRTextGenerator(llm_generator=llm, vectorizer=text_vectorizer)
ehr_gen = EHRTextGenerator(vectorizer=text_vectorizer)

# --- 2. Generate Data ---
# Generate data under the alternative hypothesis (notes depend on diagnosis)
data_alt = ehr_gen.generate_alternative(n_samples=50, dependency_strength=0.9)
# X_alt is raw text, Y_alt is diagnosis, Z_alt contains vectorized patient info
X_alt, Y_alt, Z_alt = data_alt['X'], data_alt['Y'], data_alt['Z']

# Generate data under the null hypothesis (notes independent of diagnosis given patient info)
data_null = ehr_gen.generate_null(n_samples=50)
X_null, Y_null, Z_null = data_null['X'], data_null['Y'], data_null['Z']

# --- 3. Evaluate CIT Test ---
# Initialize a CIT test method
cit_test = ConditionalRandomizationTest(n_permutations=200)

# Evaluate on alternative data (expect low p-value, reject null)
# Note: CIT methods typically require numerical input. The raw text in X
# needs to be transformed (e.g., using the vectorizer or embeddings) before passing to the test.
# Assuming a function `transform_text(X)` exists:
# p_value_alt, _ = evaluate_cit(cit_test, transform_text(X_alt), Y_alt, Z_alt)
# print(f"P-value (Alternative Hypothesis): {p_value_alt:.4f}")

# Evaluate on null data (expect high p-value, fail to reject null)
# p_value_null, _ = evaluate_cit(cit_test, transform_text(X_null), Y_null, Z_null)
# print(f"P-value (Null Hypothesis): {p_value_null:.4f}")

# You can also generate a human-readable sample:
# example_data = ehr_gen.generate_example_dataset(n_samples=2, hypothesis='alternative')
# print(example_data)
```

**Note:** When using text data (X) with most CIT methods, you need a strategy to convert the text into a numerical representation (e.g., TF-IDF vectors, sentence embeddings) suitable for the test statistic being used. The `EHRTextGenerator` example includes an optional vectorizer for this purpose.

## ArXiv Collector

Also available is `ArXivCollector` (`benchbboxtest/datasets/text/llm_text.py`), which can be used to gather text data (abstracts, potentially full papers) from arXiv for constructing CIT benchmarks.

### Usage Example (Conceptual)

```python
# (Illustrative - check the module for exact implementation details)
from benchbboxtest.datasets.text import ArXivCollector
from benchbboxtest.core import ConditionalRandomizationTest
from benchbboxtest.evaluation import evaluate_cit

# Initialize the collector, e.g., search for papers in specific categories
# collector = ArXivCollector(query='cat:cs.LG AND cat:stat.ML', max_results=200)

# Collect data (e.g., abstracts and metadata)
# collected_data = collector.collect_data()
# X = [item['abstract'] for item in collected_data] # Text data
# Y = [item['primary_category'] for item in collected_data] # Potential target variable
# Z = [(item['authors'], item['published_year']) for item in collected_data] # Potential conditioning variables

# Define a CIT task: Is the abstract content (X) independent of the primary category (Y)
# given the authors and year (Z)?

# --- Preprocess and Evaluate ---
# Preprocess X (e.g., text embeddings)
# Initialize a CIT test method
# cit_test = ConditionalRandomizationTest(...)

# Evaluate (assuming appropriate preprocessing)
# p_value, _ = evaluate_cit(cit_test, preprocess(X), Y, Z)
# print(f"P-value (ArXiv Data): {p_value:.4f}")
```

**Note:** Similar to other text datasets, the raw text collected from arXiv usually needs to be transformed into a numerical format before applying standard CIT methods. The specific way you define X, Y, and Z will depend on the research question you want to investigate using the collected papers. 