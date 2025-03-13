# setup.py
from setuptools import setup

if __name__ == "__main__":
    setup(
        install_requires=[
            "torch>=1.10.0",
            "datasets>=2.0.0",  # for dataset streaming
        ],
    )
