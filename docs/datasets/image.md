# Image Data Generation

BenchBBoxTest facilitates conditional independence testing on image data using the `CelebAMaskGenerator`.

## CelebAMaskGenerator

Located in `benchbboxtest/datasets/image/celebamask.py`, this generator utilizes the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset.

### Purpose

It allows setting up CIT problems where:
- **X**: An image, potentially with specific regions masked (e.g., masking the right eye or hair).
- **Y**: A specific facial attribute (e.g., 'Eyebrows_Visible' or 'Hair_Color').
- **Z**: Other facial attributes (e.g., 'Male', 'Young', 'Eyeglasses').

The generator aims to create scenarios based on attributes and potentially associated masks:
- **Null Hypothesis Example (from paper):** Mask the right eye region (part of X) and test if the masked image predicts the corresponding eyebrow attribute (Y) conditioned on other attributes (Z).
- **Alternative Hypothesis Example (from paper):** Mask the hair region (part of X) and test if the masked image predicts the hair color attribute (Y) conditioned on other attributes (Z).

### Usage Example

This example demonstrates the workflow for evaluating CIT methods using image data, referencing the scenarios described above. The specific masking and attribute handling are managed within the `CelebAMaskGenerator` based on the initialized `attribute`.

```python
import os
import numpy as np
from benchbboxtest.core import (
    ConditionalRandomizationTest,
    HoldoutRandomizationTest
)
from benchbboxtest.datasets.image import (
    CelebAMaskGenerator,
    download_celebamask_hq # Example helper for download
)
from benchbboxtest.evaluation import simultaneous_evaluation

# --- 1. Setup ---
np.random.seed(42)

# Define dataset directory
dataset_dir = os.path.join(os.getcwd(), "data", "celebamask-hq")
os.makedirs(dataset_dir, exist_ok=True)

# Ensure the CelebAMask-HQ dataset is downloaded and extracted here.
# The `download_celebamask_hq` is illustrative; manual download is typically required.
# if not os.path.exists(os.path.join(dataset_dir, "CelebA-HQ-img")):
#     print("Downloading CelebAMask-HQ...")
#     download_celebamask_hq(dataset_dir)

# --- 2. Initialize Generator and Tests ---
# Example: Initialize generator for the hair color prediction task (Alternative Hypothesis scenario)
print("Creating data generator for 'Blond_Hair' attribute task...")
# The generator, when generating the alternative, would handle masking the hair
# region and structuring the data accordingly.
# For the null hypothesis (eye mask -> eyebrow), you would initialize with a relevant eyebrow attribute.
hair_color_gen = CelebAMaskGenerator(dataset_path=dataset_dir, attribute="Blond_Hair") # Example attribute

# Initialize CIT test methods
crt = ConditionalRandomizationTest(n_permutations=50, random_state=42)
hrt = HoldoutRandomizationTest(n_permutations=50, test_size=0.3, random_state=42)

# --- 3. Evaluate ---
# Define sample sizes and trials
n_samples_list = [50, 100, 200]
n_trials = 5

print(f"Evaluating CRT on 'Blond_Hair' task across {n_samples_list} samples...")
# The evaluation function calls the generator's `generate_null` and `generate_alternative`,
# which implement the specific paper scenarios (e.g., eye mask for null, hair mask for alternative)
# based on the generator's internal logic tied to the initialized attribute.
results_crt = simultaneous_evaluation(
    test_method=crt,
    data_generator=hair_color_gen, # Using the generator for the hair task
    n_samples_list=n_samples_list,
    n_trials=n_trials
)

print(f"Evaluating HRT on 'Blond_Hair' task across {n_samples_list} samples...")
results_hrt = simultaneous_evaluation(
    test_method=hrt,
    data_generator=hair_color_gen, # Using the generator for the hair task
    n_samples_list=n_samples_list,
    n_trials=n_trials
)

# Results (like power curves) can then be analyzed or plotted
print("Evaluation complete. Results dictionaries contain power/type-I error estimates.")
# print(results_crt)
# print(results_hrt)

# Example plotting (requires utils.visualization):
# from benchbboxtest.utils.visualization import plot_test_comparison
# plt_comparison = plot_test_comparison({'CRT': results_crt, 'HRT': results_hrt})
# plt_comparison.show()

```

**Important Notes:**

*   **Dataset Download:** The CelebAMask-HQ dataset is large and typically requires manual download and placement in the specified `dataset_path`.
*   **Image Preprocessing:** Real-world application requires converting images (X), potentially masked, into suitable numerical representations (e.g., embeddings) before testing. The `CelebAMaskGenerator` handles or facilitates this.
*   **Conditional Variables (Z):** The specific conditional attributes (Z) used depend on the generator's implementation for the null/alternative scenarios.
*   **Masking Logic:** The exact implementation of how regions are masked and how null/alternative samples are constructed based on attributes (like predicting eyebrows from a masked eye vs. predicting hair color from masked hair) resides within the `CelebAMaskGenerator` class.
 