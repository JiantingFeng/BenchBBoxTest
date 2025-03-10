import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
import arxiv
from ...core import DataGenerator


class LLMGenerator:
    """
    Language Model Generator for text generation.
    """
    
    def __init__(self, model_name: str = 'gpt2'):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 1.0, num_return_sequences: int = 1) -> List[str]:
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
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return the generated text
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return generated_texts


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
        self.categories = categories if categories else ['cs.LG', 'cs.AI', 'stat.ML']
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
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in search.results():
            paper = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': result.published,
                'updated': result.updated,
                'categories': result.categories,
                'pdf_url': result.pdf_url
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
    
    def __init__(self, llm_generator: LLMGenerator = None, context_length: int = 50):
        """
        Initialize the text generator.
        
        Args:
            llm_generator: Language model generator
            context_length: Length of the context
        """
        self.llm_generator = llm_generator if llm_generator else LLMGenerator()
        self.context_length = context_length
    
    def generate_null(self, n_samples: int, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate data under the null hypothesis (X ⊥ Y | Z).
        
        For the null hypothesis, we generate text using only the context (Z),
        making X and Y conditionally independent given Z.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # This is a simplified implementation
        # In a real implementation, you would:
        # 1. Generate n_samples contexts (Z)
        # 2. Generate text completions (X) based only on Z
        # 3. Assign labels (Y) based on some criteria
        
        # Placeholder implementation
        contexts = [f"Context {i}" for i in range(n_samples)]
        completions = []
        labels = np.random.randint(0, 2, size=n_samples)
        
        for context in contexts:
            # Generate text based only on context
            generated_text = self.llm_generator.generate_text(context, max_length=100)[0]
            completions.append(generated_text)
        
        # Convert to numpy arrays
        X = np.array(completions)
        Y = labels.reshape(-1, 1)
        Z = np.array(contexts)
        
        return {'X': X, 'Y': Y, 'Z': Z}
    
    def generate_alternative(self, n_samples: int, **kwargs) -> Dict[str, np.ndarray]:
        """
        Generate data under the alternative hypothesis (X ⊥̸ Y | Z).
        
        For the alternative hypothesis, we generate text using both context (Z) and label (Y),
        making X and Y conditionally dependent given Z.
        
        Args:
            n_samples: Number of samples to generate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing 'X', 'Y', and 'Z' arrays
        """
        # This is a simplified implementation
        # In a real implementation, you would:
        # 1. Generate n_samples contexts (Z)
        # 2. Assign labels (Y)
        # 3. Generate text completions (X) based on both Z and Y
        
        # Placeholder implementation
        contexts = [f"Context {i}" for i in range(n_samples)]
        completions = []
        labels = np.random.randint(0, 2, size=n_samples)
        
        for i, context in enumerate(contexts):
            # Generate text based on context and label
            label = "positive" if labels[i] == 1 else "negative"
            prompt = f"{context} [Label: {label}]"
            generated_text = self.llm_generator.generate_text(prompt, max_length=100)[0]
            completions.append(generated_text)
        
        # Convert to numpy arrays
        X = np.array(completions)
        Y = labels.reshape(-1, 1)
        Z = np.array(contexts)
        
        return {'X': X, 'Y': Y, 'Z': Z} 