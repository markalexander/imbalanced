import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="imbalanced",
    version="0.0.1",
    author="Mark Alexander",
    description="Imbalanced learning add-ons for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markalexander/imbalanced",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)