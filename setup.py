from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="alzheimer-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="可解释性阿尔茨海默症多模态检测系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alzheimer-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alzheimer-train=scripts.train_end2end:main",
            "alzheimer-infer=scripts.inference:main",
            "alzheimer-preprocess=scripts.preprocess_data:main",
        ],
    },
) 