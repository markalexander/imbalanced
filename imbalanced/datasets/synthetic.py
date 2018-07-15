# -*- coding: utf-8 -*-

from sklearn.datasets import make_classification, make_regression
import numpy as np
from collections import OrderedDict
from typing import Optional, List, Union, Tuple, Any
from .generic import SimpleDataset


class SKLearnSyntheticClassification(SimpleDataset):
    """Synthetic dataset generated by scikit-learn's make_classification().

    Verbatim from the scikit-learn docs:

        Generate a random n-class classification problem.

        This initially creates clusters of points normally distributed (std=1)
        about vertices of an n_informative-dimensional hypercube with sides of
        length 2*class_sep and assigns an equal number of clusters to each
        class. It introduces interdependence between these features and adds
        various types of further noise to the data.

        Prior to shuffling, X stacks a number of these primary “informative”
        features, “redundant” linear combinations of these, “repeated”
        duplicates of sampled features, and arbitrary noise for and remaining
        features.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 20,
        n_informative: int = 2,
        n_redundant: int = 2,
        n_repeated: int = 0,
        n_classes: int = 2,
        n_clusters_per_class: int = 2,
        weights: Optional[List[float]] = None,
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        hypercube: bool = True,
        shift: Optional[Union[float, np.ndarray]] = 0.0,
        scale: Optional[Union[float, np.ndarray]] =1.0,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        """Create an SKLearnSyntheticClassification object.

        Args list verbatim from the sklearn docs.

        Args:
            n_samples:            The number of samples.
            n_features:           The total number of features. These comprise
                                  n_informative informative features,
                                  n_redundant redundant features, n_repeated
                                  duplicated features and
                                  n_features - n_informative
                                  - n_redundant - n_repeated useless features
                                  drawn at random.
            n_informative:        The number of informative features. Each class
                                  is composed of a number of gaussian clusters
                                  each located around the vertices of a
                                  hypercube in a subspace of dimension
                                  n_informative. For each cluster, informative
                                  features are drawn independently from N(0, 1)
                                  and then randomly linearly combined within
                                  each cluster in order to add covariance. The
                                  clusters are then placed on the vertices of
                                  the hypercube.
            n_redundant:          The number of redundant features. These
                                  features are generated as random linear
                                  combinations of the informative features.
            n_repeated:           The number of duplicated features, drawn
                                  randomly from the informative and the
                                  redundant features.
            n_classes:            The number of classes (or labels) of the
                                  classification problem.
            n_clusters_per_class: The number of clusters per class.
            weights:              The proportions of samples assigned to each
                                  class. If None, then classes are balanced.
                                  Note that if len(weights) == n_classes - 1,
                                  then the last class weight is automatically
                                  inferred. More than n_samples samples may be
                                  returned if the sum of weights exceeds 1.
            flip_y:               The fraction of samples whose class are
                                  randomly exchanged. Larger values introduce
                                  noise in the labels and make the
                                  classification task harder.
            class_sep:            The factor multiplying the hypercube size.
                                  Larger values spread out the clusters/classes
                                  and make the classification task easier.
            hypercube:            If True, the clusters are put on the vertices
                                  of a hypercube. If False, the clusters are put
                                  on the vertices of a random polytope.
            shift:                Shift features by the specified value. If
                                  None, then features are shifted by a random
                                  value drawn in [-class_sep, class_sep].
            scale:                Multiply features by the specified value. If
                                  None, then features are scaled by a random
                                  value drawn in [1, 100]. Note that scaling
                                  happens after shifting.
            shuffle:              Shuffle the samples and the features.
            random_state:         If int, random_state is the seed used by the
                                  random number generator; If RandomState
                                  instance, random_state is the random number
                                  generator; If None, the random number
                                  generator is the RandomState instance used by
                                  np.random.
        """
        self._args = [
            ('n_samples', n_samples),
            ('n_features', n_features),
            ('n_informative', n_informative),
            ('n_redundant', n_redundant),
            ('n_repeated', n_repeated),
            ('n_classes', n_classes),
            ('n_clusters_per_class', n_clusters_per_class),
            ('weights', weights),
            ('flip_y', flip_y),
            ('class_sep', class_sep),
            ('hypercube', hypercube),
            ('shift', shift),
            ('scale', scale),
            ('shuffle', shuffle),
            ('random_state', random_state)
        ]
        inputs, targets = make_classification(**OrderedDict(self._args))
        super().__init__(inputs, targets)

    @property
    def c_args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments ('c-args') which define
        the current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        return self._args


class SKLearnSyntheticRegression(SimpleDataset):
    """Synthetic dataset generated by scikit-learn's make_regression().

    Verbatim from the scikit-learn docs:

        Generate a random regression problem.

        The input set can either be well conditioned (by default) or have a low
        rank-fat tail singular profile. See make_low_rank_matrix for more
        details.

        The output is generated by applying a (potentially biased) random linear
        regression model with n_informative nonzero regressors to the previously
        generated input and some gaussian centered noise with some adjustable
        scale.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 100,
        n_informative: int = 10,
        n_targets: int = 1,
        bias: float = 0.0,
        effective_rank: Optional[int] = None,
        tail_strength: float = 0.5,
        noise: float = 0.0,
        shuffle: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        """Create an SKLearnSyntheticRegression object.

        Args list verbatim from the sklearn docs.


        Args:
            n_samples:      The number of samples.
            n_features:     The number of features.
            n_informative:  The number of informative features, i.e., the number
                            of features used to build the linear model used to
                            generate the output.
            n_targets:      The number of regression targets, i.e., the
                            dimension of the y output vector associated with a
                            sample. By default, the output is a scalar.
            bias:           The bias term in the underlying linear model.
            effective_rank: if not None: The approximate number of singular
                            vectors required to explain most of the input data
                            by linear combinations. Using this kind of singular
                            spectrum in the input allows the generator to
                            reproduce the correlations often observed in
                            practice.
                            if None: The input set is well conditioned, centered
                            and gaussian with unit variance.
            tail_strength:  The relative importance of the fat noisy tail of the
                            singular values profile if effective_rank is not
                            None.
            noise:          The standard deviation of the gaussian noise applied
                            to the output.
            shuffle:        Shuffle the samples and the features.
            random_state:   If int, random_state is the seed used by the random
                            number generator; If RandomState instance,
                            random_state is the random number generator; If
                            None, the random number generator is the RandomState
                            instance used by np.random.
        """
        self._args = [
            ('n_samples', n_samples),
            ('n_features', n_features),
            ('n_informative', n_informative),
            ('n_targets', n_targets),
            ('bias', bias),
            ('effective_rank', effective_rank),
            ('tail_strength', tail_strength),
            ('noise', noise),
            ('shuffle', shuffle),
            ('random_state', random_state)
        ]
        inputs, targets = make_regression(**OrderedDict(self._args))
        super().__init__(inputs, targets)

    @property
    def c_args(self) -> List[Tuple[str, Any]]:
        """Get the canonical (ordered) list of arguments ('c-args') which define
        the current object.

        Returns:
            The arguments, as a list of tuples (arg_name, arg_value).

        """
        return self._args
