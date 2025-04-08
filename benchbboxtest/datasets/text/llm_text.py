import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
import json
import arxiv
from ...core import DataGenerator


class LLMGenerator:
    """
    Language Model Generator for text generation.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the LLM generator.

        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text based on a prompt.

        Args:
            prompt: Input prompt for text generation
            max_length: Maximum length of the generated text
            temperature: Sampling temperature (higher = more random)
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated text sequences
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate text
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode and return the generated text
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts


class OpenAIGenerator:
    """
    OpenAI API Generator for text generation.

    This class provides an interface to generate text using OpenAI-compatible APIs.
    It supports both the official OpenAI API and compatible alternatives.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        organization: str = None,
    ):
        """
        Initialize the OpenAI generator.

        Args:
            api_key: OpenAI API key (if None, looks for OPENAI_API_KEY environment variable)
            base_url: Base URL for the API (can be modified for compatible APIs)
            model: Model to use for generation
            organization: OpenAI organization ID (optional)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set the OPENAI_API_KEY environment variable or pass api_key."
            )

        self.base_url = base_url
        self.model = model
        self.organization = organization

        # Determine the endpoint based on the model type
        if any(model.startswith(prefix) for prefix in ["gpt-3.5", "gpt-4"]):
            self.endpoint = f"{self.base_url}/chat/completions"
            self.is_chat_model = True
        else:
            self.endpoint = f"{self.base_url}/completions"
            self.is_chat_model = False

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        system_message: str = "You are a helpful assistant.",
        **kwargs,
    ) -> List[str]:
        """
        Generate text using the OpenAI API or compatible alternatives.

        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            num_return_sequences: Number of sequences to generate
            top_p: Nucleus sampling parameter
            stop: Sequence(s) at which to stop generation
            system_message: System message for chat models (ignored for completion models)
            **kwargs: Additional parameters to pass to the API

        Returns:
            List of generated text sequences
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # Prepare the request payload based on model type
        if self.is_chat_model:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": num_return_sequences,
                "top_p": top_p,
            }
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n": num_return_sequences,
                "top_p": top_p,
            }

        # Add stop sequences if provided
        if stop:
            payload["stop"] = stop

        # Add any additional parameters
        for key, value in kwargs.items():
            payload[key] = value

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()

            # Extract the generated text based on model type
            if self.is_chat_model:
                generated_texts = [
                    choice["message"]["content"] for choice in response_data["choices"]
                ]
            else:
                generated_texts = [
                    choice["text"] for choice in response_data["choices"]
                ]

            return generated_texts

        except requests.exceptions.RequestException as e:
            error_message = f"API request failed: {e}"

            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_message = f"API error: {error_data['error']['message']}"
                except:
                    error_message = f"API error: {e.response.text}"

            raise RuntimeError(error_message)

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get the list of available models from the API.

        Returns:
            List of model objects
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        try:
            response = requests.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
            return response.json()["data"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch models: {e}")


class ArXivCollector:
    """
    Collector for arXiv papers.
    """

    def __init__(self, categories: List[str] = None, date_from: str = None):
        """
        Initialize the arXiv collector.

        Args:
            categories: List of arXiv categories to search
            date_from: Date to start searching from (format: YYYY-MM-DD)
        """
        self.categories = categories if categories else ["cs.LG", "cs.AI", "stat.ML"]
        self.date_from = date_from

    def search_papers(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search for papers on arXiv.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of paper metadata
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in search.results():
            paper = {
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published,
                "updated": result.updated,
                "categories": result.categories,
                "pdf_url": result.pdf_url,
            }
            papers.append(paper)

        return papers

    def download_paper(self, paper_id: str, target_dir: str) -> str:
        """
        Download a paper from arXiv.

        Args:
            paper_id: arXiv paper ID
            target_dir: Directory to save the paper

        Returns:
            Path to the downloaded paper
        """
        os.makedirs(target_dir, exist_ok=True)

        # Get paper
        paper = next(arxiv.Search(id_list=[paper_id]).results())

        # Download PDF
        target_path = os.path.join(target_dir, f"{paper_id}.pdf")
        paper.download_pdf(target_path)

        return target_path


class TextGenerator(DataGenerator):
    """
    Data generator for text-based datasets.

    Implements data generation for conditional independence testing using text data.
    """

    def __init__(
        self,
        llm_generator: LLMGenerator = None,
        context_length: int = 50,
        prompt_templates: Dict[str, str] = None,
        vectorizer: Optional[Callable] = None,
    ):
        """
        Initialize the text generator.

        Args:
            llm_generator: Language model generator
            context_length: Length of the context
            prompt_templates: Dictionary of prompt templates for different scenarios
            vectorizer: Function to convert text to numerical vectors (optional)
        """
        self.llm_generator = llm_generator if llm_generator else LLMGenerator()
        self.context_length = context_length

        # Default prompt templates
        self.prompt_templates = (
            prompt_templates
            if prompt_templates
            else {
                "null_context": "Topic: {context}. Write a paragraph about this topic.",
                "alt_positive": "Topic: {context}. Write a positive paragraph about this topic.",
                "alt_negative": "Topic: {context}. Write a negative paragraph about this topic.",
            }
        )

        self.vectorizer = vectorizer

    def _generate_contexts(self, n_samples: int) -> List[str]:
        """
        Generate context topics.

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of context strings
        """
        # Topics from diverse domains
        topics = [
            "Artificial Intelligence",
            "Climate Change",
            "Renewable Energy",
            "Quantum Computing",
            "Blockchain Technology",
            "Space Exploration",
            "Genetic Engineering",
            "Virtual Reality",
            "Robotics",
            "Cybersecurity",
            "Nanotechnology",
            "Internet of Things",
            "Autonomous Vehicles",
            "Machine Learning",
            "Social Media",
            "Biotechnology",
            "3D Printing",
            "Digital Privacy",
            "Cloud Computing",
            "Sustainable Development",
        ]

        # Sample topics with replacement if n_samples > len(topics)
        return np.random.choice(
            topics, size=n_samples, replace=(n_samples > len(topics))
        ).tolist()

    def generate_null(
        self, n_samples: int, temperature: float = 1.0, max_length: int = 150
    ) -> Dict[str, np.ndarray]:
        """
        Generate data under the null hypothesis (X ⊥ Y | Z).

        For the null hypothesis, we generate text using only the context (Z),
        making X and Y conditionally independent given Z.

        Args:
            n_samples: Number of samples to generate
            temperature: Control randomness in generation (higher = more random)
            max_length: Maximum length of generated text

        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # Generate contexts (Z)
        contexts = self._generate_contexts(n_samples)

        # Generate completions (X) based only on context
        completions = []

        for context in contexts:
            # Create prompt using the null context template
            prompt = self.prompt_templates["null_context"].format(context=context)

            # Generate text based only on context
            generated_text = self.llm_generator.generate_text(
                prompt, max_length=max_length, temperature=temperature
            )[0]

            completions.append(generated_text)

        # Assign binary labels (Y) randomly and independently of X given Z
        # This ensures conditional independence (X ⊥ Y | Z)
        labels = np.random.randint(0, 2, size=n_samples)

        # Convert to numpy arrays
        X = np.array(completions)
        Y = labels.reshape(-1, 1)
        Z = np.array(contexts)

        # Apply vectorization if provided
        if self.vectorizer is not None:
            X = self.vectorizer(X)

        return {"X": X, "Y": Y, "Z": Z}

    def generate_alternative(
        self,
        n_samples: int,
        dependency_strength: float = 0.8,
        temperature: float = 0.8,
        max_length: int = 150,
    ) -> Dict[str, np.ndarray]:
        """
        Generate data under the alternative hypothesis (X ⊥̸ Y | Z).

        For the alternative hypothesis, we generate text using both context (Z) and label (Y),
        making X and Y conditionally dependent given Z.

        Args:
            n_samples: Number of samples to generate
            dependency_strength: Controls how strongly Y influences X (0.0-1.0)
            temperature: Control randomness in generation (higher = more random)
            max_length: Maximum length of generated text

        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # Generate contexts (Z)
        contexts = self._generate_contexts(n_samples)

        # Assign binary labels (Y) randomly
        labels = np.random.randint(0, 2, size=n_samples)

        # Generate completions (X) based on both context and label
        completions = []

        for i, context in enumerate(contexts):
            # Only use label information with probability = dependency_strength
            # This controls how strong the dependency between X and Y is
            use_label = np.random.random() < dependency_strength

            if use_label:
                # Create prompt using either positive or negative template based on the label
                template_key = "alt_positive" if labels[i] == 1 else "alt_negative"
                prompt = self.prompt_templates[template_key].format(context=context)
            else:
                # Occasionally use null template to make dependency less deterministic
                prompt = self.prompt_templates["null_context"].format(context=context)

            # Generate text based on context and possibly label
            generated_text = self.llm_generator.generate_text(
                prompt, max_length=max_length, temperature=temperature
            )[0]

            completions.append(generated_text)

        # Convert to numpy arrays
        X = np.array(completions)
        Y = labels.reshape(-1, 1)
        Z = np.array(contexts)

        # Apply vectorization if provided
        if self.vectorizer is not None:
            X = self.vectorizer(X)

        return {"X": X, "Y": Y, "Z": Z}


class EHRTextGenerator(DataGenerator):
    """
    Data generator for Electronic Health Records (EHR) text-based datasets.

    Implements data generation for conditional independence testing using EHR data,
    where:
    - X: Clinical notes or medical reports
    - Y: Patient outcomes or diagnoses (binary)
    - Z: Patient demographics and medical history
    """

    def __init__(
        self,
        llm_generator: LLMGenerator = None,
        vectorizer: Optional[Callable] = None,
    ):
        """
        Initialize the EHR text generator.

        Args:
            llm_generator: Language model generator
            vectorizer: Function to convert text to numerical vectors (optional)
        """
        self.llm_generator = llm_generator if llm_generator else LLMGenerator()
        self.vectorizer = vectorizer

        # Templates for generating EHR clinical notes
        self.templates = {
            # Templates for null hypothesis
            "null": {
                "note_template": "Patient demographics: {demographics}\nMedical history: {history}\n\nClinical Notes:",
            },
            # Templates for alternative hypothesis (influenced by outcome/diagnosis)
            "alt_positive": {
                "note_template": "Patient demographics: {demographics}\nMedical history: {history}\nDiagnosis: {diagnosis}\n\nClinical Notes:",
            },
            "alt_negative": {
                "note_template": "Patient demographics: {demographics}\nMedical history: {history}\nNo evidence of: {diagnosis}\n\nClinical Notes:",
            },
        }

        # Example demographics, medical histories, and diagnoses
        self.demographics = [
            "45-year-old male",
            "62-year-old female",
            "37-year-old female",
            "53-year-old male",
            "78-year-old female",
            "29-year-old male",
            "41-year-old female",
            "67-year-old male",
            "31-year-old female",
            "59-year-old male",
        ]

        self.medical_histories = [
            "Hypertension, Type 2 diabetes",
            "Asthma, Seasonal allergies",
            "Hypercholesterolemia, Osteoarthritis",
            "Coronary artery disease, Prior myocardial infarction",
            "COPD, Osteoporosis",
            "No significant past medical history",
            "Hypothyroidism, Depression",
            "Rheumatoid arthritis, Chronic kidney disease",
            "Migraines, Anxiety disorder",
            "Atrial fibrillation, Heart failure",
        ]

        self.diagnoses = [
            "Pneumonia",
            "Acute myocardial infarction",
            "Urinary tract infection",
            "Community-acquired pneumonia",
            "Congestive heart failure exacerbation",
            "Acute pancreatitis",
            "Diabetic ketoacidosis",
            "Ischemic stroke",
            "Cellulitis",
            "Acute appendicitis",
        ]

    def _generate_patient_data(self, n_samples: int) -> Tuple[List[str], List[str]]:
        """
        Generate patient demographics and medical history.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple containing lists of demographics and medical histories
        """
        # Sample with replacement if n_samples > len(demographics/medical_histories)
        demographics = np.random.choice(
            self.demographics,
            size=n_samples,
            replace=(n_samples > len(self.demographics)),
        ).tolist()

        histories = np.random.choice(
            self.medical_histories,
            size=n_samples,
            replace=(n_samples > len(self.medical_histories)),
        ).tolist()

        return demographics, histories

    def generate_null(
        self, n_samples: int, temperature: float = 0.7, max_length: int = 200
    ) -> Dict[str, np.ndarray]:
        """
        Generate EHR data under the null hypothesis (X ⊥ Y | Z).

        For the null hypothesis, clinical notes (X) are generated based only on
        patient demographics and medical history (Z), independent of diagnosis (Y).

        Args:
            n_samples: Number of samples to generate
            temperature: Control randomness in generation
            max_length: Maximum length of generated text

        Returns:
            Dictionary containing 'X' (clinical notes), 'Y' (diagnoses), and 'Z' (patient info)
        """
        # Generate patient demographics and medical history (Z)
        demographics, histories = self._generate_patient_data(n_samples)

        # Combine demographics and histories to form Z
        Z_data = [f"{dem}; {hist}" for dem, hist in zip(demographics, histories)]

        # Generate clinical notes (X) based only on demographics and history
        clinical_notes = []

        for i in range(n_samples):
            # Create prompt using the null template
            prompt = self.templates["null"]["note_template"].format(
                demographics=demographics[i], history=histories[i]
            )

            # Generate clinical note text based on patient info
            generated_note = self.llm_generator.generate_text(
                prompt, max_length=max_length, temperature=temperature
            )[0]

            clinical_notes.append(generated_note)

        # Assign diagnoses (Y) randomly and independently of X given Z
        # Note: Binary classification (1: diagnosed, 0: not diagnosed)
        diagnoses = np.random.randint(0, 2, size=n_samples)

        # Convert to numpy arrays
        X = np.array(clinical_notes)
        Y = diagnoses.reshape(-1, 1)
        Z = np.array(Z_data)

        # Apply vectorization if provided
        if self.vectorizer is not None:
            X = self.vectorizer(X)

        return {"X": X, "Y": Y, "Z": Z}

    def generate_alternative(
        self,
        n_samples: int,
        dependency_strength: float = 0.9,
        temperature: float = 0.7,
        max_length: int = 200,
    ) -> Dict[str, np.ndarray]:
        """
        Generate EHR data under the alternative hypothesis (X ⊥̸ Y | Z).

        For the alternative hypothesis, clinical notes (X) are influenced by both
        patient info (Z) and diagnosis outcome (Y), making X and Y conditionally dependent.

        Args:
            n_samples: Number of samples to generate
            dependency_strength: Controls how strongly diagnosis influences notes (0.0-1.0)
            temperature: Control randomness in generation
            max_length: Maximum length of generated text

        Returns:
            Dictionary containing 'X' (clinical notes), 'Y' (diagnoses), and 'Z' (patient info)
        """
        # Generate patient demographics and medical history (Z)
        demographics, histories = self._generate_patient_data(n_samples)

        # Combine demographics and histories to form Z
        Z_data = [f"{dem}; {hist}" for dem, hist in zip(demographics, histories)]

        # Assign diagnoses (Y) randomly
        # Note: Binary classification (1: diagnosed, 0: not diagnosed)
        diagnoses = np.random.randint(0, 2, size=n_samples)

        # Sample specific diagnoses from the list
        specific_diagnoses = np.random.choice(
            self.diagnoses, size=n_samples, replace=(n_samples > len(self.diagnoses))
        ).tolist()

        # Generate clinical notes (X) based on patient info and diagnosis
        clinical_notes = []

        for i in range(n_samples):
            # Use diagnosis information with probability = dependency_strength
            # This controls how strong the dependency between notes and diagnosis is
            use_diagnosis = np.random.random() < dependency_strength

            if use_diagnosis:
                # Use template that includes diagnosis information
                template_key = "alt_positive" if diagnoses[i] == 1 else "alt_negative"
                prompt = self.templates[template_key]["note_template"].format(
                    demographics=demographics[i],
                    history=histories[i],
                    diagnosis=specific_diagnoses[i],
                )
            else:
                # Occasionally use null template to make dependency less deterministic
                prompt = self.templates["null"]["note_template"].format(
                    demographics=demographics[i], history=histories[i]
                )

            # Generate clinical note
            generated_note = self.llm_generator.generate_text(
                prompt, max_length=max_length, temperature=temperature
            )[0]

            clinical_notes.append(generated_note)

        # Convert to numpy arrays
        X = np.array(clinical_notes)
        Y = diagnoses.reshape(-1, 1)
        Z = np.array(Z_data)

        # Apply vectorization if provided
        if self.vectorizer is not None:
            X = self.vectorizer(X)

        return {"X": X, "Y": Y, "Z": Z}

    def generate_example_dataset(
        self, n_samples: int = 5, hypothesis: str = "alternative"
    ) -> Dict[str, List]:
        """
        Generate a small example dataset with human-readable format.

        Args:
            n_samples: Number of samples to generate
            hypothesis: 'null' or 'alternative'

        Returns:
            Dictionary with human-readable examples
        """
        if hypothesis.lower() == "null":
            data = self.generate_null(n_samples, temperature=0.7)
        else:
            data = self.generate_alternative(n_samples, temperature=0.7)

        # Convert numpy arrays to lists for readability
        examples = {
            "clinical_notes": data["X"].tolist(),
            "diagnoses": data["Y"].reshape(-1).tolist(),
            "patient_info": data["Z"].tolist(),
            "hypothesis": hypothesis,
        }

        return examples
