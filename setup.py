from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).parent

setup(
    name="vibe_coding",
    version="0.1.0",
    description="VIBE ML project",
    author="w.vibe",
    packages=find_packages(where="src"),   # 安装 src 下的所有包
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        # ---- runtime dependencies (别人 pip install vibe_coding 时需要) ----
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "albumentations>=1.0.0",
        "opencv-python-headless>=4.9.0",
        "transformers>=4.0.0",
        "datasets>=2.0.0",
        "python-dotenv>=1.0.0",
        "scikit-learn>=1.0.0",
        "requests>=2.0.0",
        "tqdm>=4.65.0",
        "boto3>=1.37.0",
        "wandb>=0.16.0",
        "tensorboard>=2.19.0",
    ],
)
