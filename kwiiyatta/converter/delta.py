from nnmnkwii.preprocessing import delta_features

import numpy as np

from . import abc


DELTA_WINDOWS = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]


class DeltaFeatureDataset(abc.MapDataset):
    with_key = True
    with_raw = True

    def __init__(self, base):
        super().__init__(base)
        self.frame_period = None

    def function(self, feature, raw, key):
        if self.frame_period is None:
            self.frame_period = raw.frame_period
        elif self.frame_period != raw.frame_period:
            raise ValueError(f'frame_period of "{key}" is {raw.frame_period!r}'
                             f' but others are {self.frame_period!r}')

        return delta_features(feature, DELTA_WINDOWS)


class DeltaFeatureConverter(abc.MapFeatureConverter):
    def train(self, dataset, keys, **kwargs):
        delta_dataset = DeltaFeatureDataset(dataset)
        self.base.train(delta_dataset, keys, **kwargs)
        self.frame_period = delta_dataset.frame_period

    def convert(self, feature, raw, **kwargs):
        if self.frame_period != raw.frame_period:
            raise ValueError(f'frame_period is expected to'
                             f' {self.frame_period!s}'
                             f' but {raw.frame_period!s}')

        dim = feature.shape[-1]
        result = super().convert(delta_features(feature, DELTA_WINDOWS),
                                 **kwargs)
        if result.shape[-1] > dim:
            result = result[:, :dim]
        return result
