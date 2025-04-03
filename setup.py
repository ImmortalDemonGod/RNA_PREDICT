# setup.py
from setuptools import find_packages, setup

setup(
    name="rna_predict",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0,<2.6.0",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
        "scipy>=1.7.0",
        "biopython>=1.81",
        "pandas>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
    },
    python_requires=">=3.8",
)
