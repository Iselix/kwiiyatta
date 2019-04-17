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
