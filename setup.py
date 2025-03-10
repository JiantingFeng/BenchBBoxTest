from setuptools import setup, find_packages

setup(
    name="benchbboxtest",
    version="0.1.0",
    description="Benchmarking Datasets for Black-Box Conditional Independence Testing",
    author="Jianting Feng",
    author_email="fengjianting@hotmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "torch",
        "pillow",
        "transformers",  # For text-based datasets using LLMs
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
