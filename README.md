
# Imbalanced

Imbalanced learning tools and experimental framework for PyTorch.
Big data friendly.

**N.B. this is pre-alpha and not yet functioning software.**


## Getting started

Install the `imbalanced` package with pip:

    pip install imbalanced

Train and use a pipeline with SMOTE and re-calibration:

    import imbalanced as imb

    pipeline = imb.Pipeline(
        imb.preprocessors.SMOTE(),
        net,
        imb.postprocessors.BinnedCalibrator()
    )

    # Train the net and calibrator
    pipeline.train(dataset)

    # Get some predictions
    predictions = pipeline.predict(inputs)

Where `dataset` is your PyTorch dataset, and `net` is a net module object.


## Why not the original `imbalanced-learn` package?

This package focuses on neural networks (particularly deep architectures) and
aims to be a big-data-friendly implementation in terms of storage and memory
use.
