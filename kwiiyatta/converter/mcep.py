import copy

import numpy as np

from . import abc


class MelCepstrumDataset(abc.MapDataset):
    with_key = True

    def __init__(self, base):
        super().__init__(base)
        self.fs = None
        self.order = None

    def function(self, feature, key):
        if self.order is None:
            self.order = feature.mel_cepstrum_order
        elif self.order != feature.mel_cepstrum_order:
            raise ValueError(f'order of "{key}" is'
                             f' {feature.mel_cepstrum_order!s}'
                             f' but others are {self.order!s}')

        if self.fs is None:
            self.fs = feature.fs
        elif self.fs != feature.fs:
            raise ValueError(f'fs of "{key}" is {feature.fs!s}'
                             f' but others are {self.fs!s}')

        return feature.mel_cepstrum.data[:, 1:]  # Drop 1st (power) dimension


class MelCepstrumFeatureConverter(abc.MapFeatureConverter):
    def train(self, dataset, keys, **kwargs):
        mcep_dataset = MelCepstrumDataset(dataset)
        self.base.train(mcep_dataset, keys, **kwargs)
        self.order = mcep_dataset.order
        self.fs = mcep_dataset.fs

    def convert(self, mel_cepstrum, **kwargs):
        if self.order != mel_cepstrum.order:
            raise ValueError(f'order is expected to {self.order!s}'
                             f' but {mel_cepstrum.order!s}')

        if self.fs != mel_cepstrum.fs:
            raise ValueError(f'fs is expected to {self.fs!s}'
                             f' but {mel_cepstrum.fs!s}')

        result = copy.copy(mel_cepstrum)
        result.data = np.hstack((
            result.data[:, 0].reshape(-1, 1),
            super().convert(result.data[:, 1:], raw=mel_cepstrum, **kwargs)))

        return result
