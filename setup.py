from setuptools import find_packages, setup

setup(
    name="vibe_coding",
    version="0.1.0",
    description="VIBE ML project",
    author="w.vibe",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        # Core ML libraries
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        # Computer vision
        "pillow>=10.0.0",
        "albumentations>=1.0.0",
        "opencv-python>=4.9.0",
        # Transformers/NLP
        "transformers>=4.0.0",
        "datasets>=2.0.0",
        # Utilities
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.0.0",
        "requests>=2.0.0",
        # Logging and experiment tracking
        "wandb>=0.16.0",
        # Development and testing
        "pytest>=7.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
    ],
)
