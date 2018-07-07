
# Imbalanced

Imbalanced learning tools and experimental framework for PyTorch.
Big data friendly.

**This is pre-alpha software.  Features and the exposed API are likely to
change.**


## Getting started

Python >= 3.5 is required.

Install the package with pip:

    pip install git+https://github.com/markalexander/imbalanced.git#egg=imbalanced


Automatically choose a reasonable pipeline, train it, and predict:

```python
import imbalanced as imb

# Pick a dataset
dataset = imb.datasets.SKLearnSyntheticRegression()

# Partition it into train/val/test
dataset = dataset.partitioned()

# Train and test a pipeline
pipeline = imb.AutoPipeline(dataset)
pipeline.train(dataset.train)
predictions = pipeline.predict(dataset.test)
```

Manually set up a pipeline with random subsampling and no calibration:

```python
pipeline = imb.Pipeline(
    imb.preprocessors.RandomSubsampler(rate=0.5),
    imb.nets.ExampleMLP()  # Or any PyTorch net module
)
pipeline.train(dataset.train)
predictions = pipeline.predict(dataset.train)
```


## Documentation

The code is generally self-documenting.  Web documentation will be generated
for the first full release.


## FAQ

### **Why not the existing `imbalanced-learn` package?**

This package focuses on neural networks (particularly deep architectures) and
aims to be a big-data-friendly implementation in terms of storage and memory
use.

There is also slightly more flexibility for specialized methods in the exposed
APIs.

### **Why no Python 2.7 support?**

Python 2.7 is due to be sunset at the end of 2019, with many important and
widely-used libraries dropping support before then.  For this reason, I haven't
put any extra effort into supporting it either.  Apologies if this affects you
in a way you can't get around.  I would *guess* that the package is roughly
compatible however, and an automated tool (to remove e.g. type hints and other
such things) would probably do a reasonable job of making it work on 2.7.
