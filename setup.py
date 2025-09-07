"""Setup script for Associative Memory Models.

This setup.py provides compatibility for users who prefer pip over Poetry or Guix.
Install with: pip install .
Install with dev dependencies: pip install .[dev]
Install with all optional features: pip install .[all]
Install for specific use cases: pip install .[audio] or .[video] or .[graph]
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="associative",
    version="0.1.0",
    author="Ayan Das",
    author_email="bvits@riseup.net",
    description="PyTorch implementation of associative memory models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/b-vitamins/associative",
    project_urls={
        "Bug Tracker": "https://github.com/b-vitamins/associative/issues",
        "Documentation": "https://github.com/b-vitamins/associative#readme",
        "Source Code": "https://github.com/b-vitamins/associative",
    },
    license="MIT",
    keywords=[
        "pytorch",
        "associative-memory",
        "hopfield-networks",
        "energy-based-models",
        "deep-learning",
        "transformers",
        "multimodal-learning",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "associative": ["py.typed"],  # Include type hints
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        # Core ML frameworks
        "torch>=2.0.0,<3.0.0",  # PyTorch 2.8.0 in Guix
        "torchvision>=0.15.0,<1.0.0",  # 0.23.0 in Guix
        "torch-geometric>=2.3.0,<3.0.0",  # 2.7.0 in Guix
        "torchaudio>=2.0.0,<3.0.0",  # 2.8.0 in Guix
        "torchmetrics>=1.0.0,<2.0.0",  # 1.4.1 in Guix
        # Note: torchvggish has no specific version requirement
        # Transformers and utilities
        "transformers>=4.40.0,<5.0.0",  # 4.44.2 in Guix
        "huggingface-hub>=0.20.0,<1.0.0",  # 0.32.2 in Guix
        "einops>=0.6.0,<1.0.0",  # 0.8.1 in Guix
        # Data processing
        "pandas>=2.0.0,<3.0.0",  # 2.2.3 in Guix
        "pyarrow>=20.0.0,<22.0.0",  # 21.0.0 in Guix
        # Audio processing
        "librosa>=0.10.0,<1.0.0",  # 0.10.2.post1 in Guix
        "soundfile>=0.12.0,<1.0.0",  # 0.13.1 in Guix
        "pesq>=0.0.4,<1.0.0",  # 0.15 in Guix
        "pystoi>=0.3.0,<1.0.0",  # 0.4.1 in Guix
        # Video processing
        "decord>=0.6.0,<1.0.0",  # 0.6.0 in Guix
        # Visualization
        "pillow>=10.0.0,<12.0.0",  # 11.1.0 in Guix
        "matplotlib>=3.5.0,<4.0.0",  # 3.8.2 in Guix
        "seaborn>=0.13.0,<1.0.0",  # 0.13.2 in Guix
        "scikit-learn>=1.0.0,<2.0.0",  # 1.6.1 in Guix
        # General utilities
        "tqdm>=4.60.0,<5.0.0",  # 4.67.1 in Guix
        "hydra-core>=1.3.0,<2.0.0",  # 1.3.2 in Guix
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0,<9.0.0",  # 8.3.3 in Guix
            "pytest-cov>=6.0.0,<7.0.0",  # 6.2.1 in Guix
            "ruff>=0.9.0,<1.0.0",  # 0.9.5 in Guix
        ],
        "audio": [
            # Additional audio dependencies if needed
            "torchvggish",
        ],
        "video": [
            # Video-specific dependencies
            "decord>=0.6.0",
        ],
        "graph": [
            # Graph neural network dependencies
            "torch-geometric>=2.3.0",
        ],
        "all": [
            # Include all optional dependencies
            "torchvggish",
            "decord>=0.6.0",
            "torch-geometric>=2.3.0",
        ],
    },
)
