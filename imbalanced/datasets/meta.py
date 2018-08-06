# -*- coding: utf-8 -*-


class ImbalancedDatasetStats:

    def __init__(self, dataset):
        self.dataset = dataset

        targets = dataset.tensors[1].numpy()
        self.len = len(dataset)
        self.majority_count = len(np.where(targets == 0))
        self.minority_count = len(dataset) - self.majority_count
        self.majority_ratio = self.majority_count / self.minority_count
        self.minority_ratio = self.minority_count / self.majority_count

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return 'Total count: {}, maj. count: {}, min. count: {}, maj ratio: {}, min. ratio: {}'.format(
            self.len,
            self.majority_count,
            self.minority_count,
            self.majority_ratio,
            self.minority_ratio
        )