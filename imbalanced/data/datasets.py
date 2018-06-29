# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader


class Dataset(TorchDataset):
    """Base class for all datasets, with common functionality."""

    def loader(self, *args, **kwargs):
        """Get a torch DataLoader object for this dataset.

        :param args:
        :param kwargs:
        :return:
        """
        return DataLoader(self, *args, **kwargs)

    @abstractmethod
    def __getitem__(self, idx):
        """Get a data row by index.

        :param idx:  the index of the desired row
        :type  idx:  int
        :return:     the desired row, if it exists
        :rtype:      torch.Tensor
        """
        pass

    @abstractmethod
    def __len__(self):
        """Get the total number of rows in the dataset.

        :return:  the number of rows
        :rtype:   int
        """
        pass


class DatasetWrapper(Dataset):
    """Base class for datasets that come from wrapping another dataset object.

    E.g. resamplers or other pre-processors.
    """

    def __init__(self, dataset):
        self._dataset = None
        self.dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        assert isinstance(dataset, Dataset),\
            'Wrapped object must be an instance of Dataset (or a subclass).'
        self._dataset = dataset

    @abstractmethod
    def __getitem__(self, idx):
        """Get a data row by index.

        :param idx:  the index of the desired row
        :type  idx:  int
        :return:     the desired row, if it exists
        :rtype:      torch.Tensor
        """
        pass

    @abstractmethod
    def __len__(self):
        """Get the total number of rows in the dataset.

        :return:  the number of rows
        :rtype:   int
        """
        pass


class SimpleDataset(Dataset):
    """Simple dataset from common tensor-like objects.

    Allows common tensor-like objects e.g. numpy arrays, list-of-lists to be
    augmented with Dataset's interface and used in processes that require said
    interface.
    """

    def __init__(self, inputs, targets):
        """Create a SimpleDataset object.

        :param inputs:   input features
        :type  inputs:   np.ndarray or any object that can be converted to one
                         using np.array(object)
        :param targets:  target values
        :type  targets:  np.ndarray or any object that can be converted to one
                         using np.array(object)
        """
        # Convert everything else to np.ndarray first
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
        # Validation
        assert len(inputs) == len(targets),\
            'Inputs and targets arrays must have same length'
        # Create tensor objects
        self._inputs = torch.from_numpy(inputs)
        self._targets = torch.from_numpy(targets)

    def __getitem__(self, idx):
        """Get a data row by index.

        :param idx:  the index of the desired row
        :type  idx:  int
        :return:     the desired row, if it exists
        :rtype:      torch.Tensor
        """
        return self._inputs[idx], self._targets[idx]

    def __len__(self):
        """Get the total number of rows in the dataset.

        :return:  the number of rows
        :rtype:   int
        """
        return len(self._inputs)


class MaskedDataset(DatasetWrapper):
    """Dataset wrapper that masks a given dataset."""

    def __init__(self, dataset, mask=None):
        super().__init__(dataset)
        self._active_rows = None
        if mask is not None:
            self.mask(mask)
        else:
            # Default to all rows active
            self.mask(np.ones((len(dataset),), dtype=np.bool))

    def mask(self, mask):
        """Apply the given row mask to the dataset.

        :param mask:  the mask
        :type  mask:  numpy.ndarray
        :return:
        """
        assert isinstance(mask, np.ndarray),\
            'Mask object must be numpy ndarray'
        assert len(mask) == len(self.dataset),\
            'Mask length must match dataset length'
        # Flatten and recheck
        # If length differs now then it was too wide to begin with
        mask = mask.flatten()
        assert len(mask) == len(self.dataset),\
            'Mask must be flat or 1-dimensional'
        self._active_rows = np.nonzero(mask)[0]

    def __getitem__(self, idx):
        """Get a data row by index.

        :param idx:  the index of the desired row
        :type  idx:  int
        :return:     the desired row, if it exists
        :rtype:      torch.Tensor
        """
        return self.dataset[self._active_rows[idx]]

    def __len__(self):
        """Get the total number of rows in the dataset.

        :return:  the number of rows
        :rtype:   int
        """
        return len(self._active_rows)
