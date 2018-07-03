
# Imbalanced

Imbalanced learning tools and experimental framework for PyTorch.
Big data friendly.

**N.B. this is pre-alpha and not yet functioning software.**


## Getting started

Python >= 3.5 is required.

<!--Install the `imbalanced` package with pip:

    pip install imbalanced

Train and use a pipeline with SMOTE and re-calibration:

    import imbalanced as imb

    pipeline = imb.Pipeline(
        imb.preprocessors.SMOTE(),
        imb.net,
        imb.postprocessors.BinnedCalibrator()
    )

    # Train the net and calibrator
    pipeline.train(dataset)

    # Get some predictions
    predictions = pipeline.predict(inputs)

Where `dataset` is your PyTorch dataset, and `net` is a net module object.
-->

## FAQ

### **Why not the existing `imbalanced-learn` package?**

This package focuses on neural networks (particularly deep architectures) and
aims to be a big-data-friendly implementation in terms of storage and memory
use.

There is also slightly more flexibility for specialized methods in the exposed
APIs.

### **Why no Python 2.7 support?**

Python 2.7 is due to be sunset at the start of 2020, with many important and
widely-used libraries dropping support before then.  For this reason, I haven't
put any extra effort into supporting it either.  Apologies if this affects you
in a way you can't get around.  I would *guess* that the package is roughly
compatible however, and an automated tool (to remove e.g. type hints and other
such things) would probably do a reasonable job of making it work on 2.7.
