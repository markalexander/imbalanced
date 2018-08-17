
# Imbalanced

Imbalanced learning tools and experimental framework for PyTorch.
Big data friendly.

<!--REPORT-IGNORE-->
**This is pre-alpha software.  Not all planned features are implemented.
Features and the exposed API are likely to change.**

----
<!--/REPORT-IGNORE-->


## Quick Start

Python >= 3.5 is required.

Install the package with pip:

    pip install git+https://github.com/markalexander/imbalanced.git#egg=imbalanced

<!--REPORT-IGNORE-->
Automatically choose a reasonable pipeline, train it, and test:

```python
    import imbalanced as imb
    
    # Pick a dataset
    dataset = imb.datasets.SKLearnSyntheticRegression()
    
    # Partition it into train/val/test
    dataset = dataset.partitioned()
    
    # Train and test a pipeline
    pipeline = imb.AutoPipeline(dataset)
    pipeline.train_all(dataset.train, dataset.val)
    predictions = pipeline.predict(dataset.test)
```

Manually set up a pipeline with random subsampling and no calibration:

```python
    pipeline = imb.Pipeline(
        imb.preprocessors.RandomSubsampler(rate=0.5),
        imb.nets.ExampleMLP()  # Or any PyTorch net module
    )
    pipeline.train(dataset.train, dataset.val)
    print(MSELoss(
        pipeline.predict(dataset.test),
        dataset.test.targets
    ))
    predictions = pipeline.predict(dataset.test)

```
<!--/REPORT-IGNORE-->

## Datasets and Wrappers

Generally speaking, any object which inherits from PyTorch's `Dataset` can be
used with this package.

Most dataset classes provided here are actually wrappers for other datasets.
These wrappers can often be though of as *applying* something to the wrapped
dataset.  For example, `ResampledDataset` takes a sampler or indices
and applies this to resample the underlying dataset.  So the following code:

    dataset = MyDataset()
    print('Size before: {}'.format(len(dataset)))
    dataset = ResampledDataset(dataset, RandomResampler(0.5))
    print('Size after: {}'.format(len(dataset)))

gives:

    Length before: 100
    Length after: 50

Note that after wrapping, the new object acts exactly as a normal dataset and
can be used in all of the same contexts, including recursive wrapping.

The wrapping pattern is ubiquitous in this package because it is an approach
that scales well to larger datasets.  Rather than creating a whole new and
separate dataset whenever minor changes are required, the wrapper simply stores
the *differences* between the two and serves up rows based on ad hoc logic. 
This method is much more memory efficient than the alternative, especially
when the wrapper is able to take advantage of sparsity.

Wrappers can also provide more general functionality or convenient shortcuts.
For example, you can use `PartitionedDataset` to easily wrap and partition a
given dataset into your chosen train-val-test split.

For more information, see the `imbalanced.datasets` sub-package.  Highlights
include:

  - `ResampledDataset`, for resampling datasets.  See also the 'Samplers'
    section below.
  - `TransformedDataset`, for applying transformations to the wrapped dataset.
    Useful, for example, in creating multi-stage models where the output of one
    model is the input of the next--simply wrap the original dataset with a
    transformation using the first predictor and use the new dataset directly
    with the next predictor.
  - `BinnedDataset`, for binning target values of a dataset, yielding a new one
    where the classes represent target values grouped by bins.  See also the
    'Binners' section below.  Used in multi-stage/reconstructed regressor
    models.


## Pipelines

If a dataset defines an imbalanced learning task, then a pipeline defines a
solution.  Here, `Pipeline` objects consist of a predictor of some kind, and
an optional sampler object.  Pipelines can be trained through convenience
functions or by training their components directly.  Once trained, pipelines
can make end-to-end predictions through their `forward()` or `__call__`
methods.


## Samplers

Resampling is a common method for imbalanced data problems where the dataset,
before being used for training, is sub- or super-sampled in some manner.
Common resamplers are defined here and can be combined with the
`ResampledDataset` wrapper as above to provide a resampled version of any
dataset object.  Note that samplers such as `RandomResampler`, and
`RandomTargetedResampler` are capable of doing both sub- and super-sampling, 
depending on whether the passed `rate` value is greater or less and one.  See
`samplers.py` for more information.


## Predictors

Predictors are the second main component of pipelines.  These can come in many
forms, but in general they accept input features and output predictions.
Generally, the predictors here consist of neural network models or some
combination thereof.

### Classifiers

  - `FeedForwardClassifier` - A simple MLP neural classifier.  N.B. uses
    `log_softmax`, so make sure to use the appropriate loss function (typically
    `NLLLoss`).
  - `TemperatureScaledClassifier` - A calibrator that, when trained, will be
    calibrated using temperature scaling.


### Regressors

  - `FeedForwardRegressor` - The regressor paralell to `FeedForwardClassifier`.
  - `HurdleRegressor` - A hurdle model regressor.  First predicts whether a
    given set of inputs belongs to the minority class or the majority class,
    then uses a separate regressor for each class to obtain the real-valued
    prediction.
  - `ReconRegressor` - A reconstructed regressor model.  A two-stage model that
    first trains and calibrates a classifier on a binned classification version
    of the dataset, and then uses the outputs of that classifier to reconstruct
    a point estimate.


## Metrics

As well as the standard collection of PyTorch metrics and loss functions,
some imbalanced data-specific alternatives are also provided here.

  - `ECELoss` - Estimated Calibration Loss


## FAQ

### Why not the existing `imbalanced-learn` package?

This package differs from `imbalanced-learn` in a number of ways:

  - PyTorch integration.
  - A focus on neural networks (particularly deep architectures).
  - Big-data-friendly implementation in terms of:
    - Storage and memory considerations.
    - Use of CUDA not only for the net operations, but also for e.g. faster
      pre- and post-processing, where appropriate.
  - Slightly more flexibility for specialized methods in the exposed APIs.
  
If you don't need these, then you might want to use `imbalanced-learn` instead.


### Why no Python 2.7 support?

Python 2.7 is due to be sunset at the end of 2019, with many important and
widely-used libraries dropping support before then.  For this reason, I haven't
put any extra effort into supporting it either.  Apologies if this affects you
in a way you can't get around.  I would *guess* that the package is roughly
compatible however, and an automated tool (to remove e.g. type hints and other
such things) would probably do a reasonable job of making it work on 2.7.


### Are there unit tests?

Yes.  These can be discovered and run using pytest.