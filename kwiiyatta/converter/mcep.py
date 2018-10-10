import copy

import numpy as np

import kwiiyatta

from . import abc


class MelCepstrumDataset(abc.MapDataset):
    with_key = True

    def __init__(self, base, mcep_fs=None):
        super().__init__(base)
        self.fs = mcep_fs
        self.order = None

    def function(self, feature, key):
        f = kwiiyatta.feature(feature)

        if self.order is None:
            self.order = f.mel_cepstrum_order
        elif self.order != feature.mel_cepstrum_order:
            f.mel_cepstrum_order = self.order

        mcep_data = f.mel_cepstrum.data

        if self.fs is None:
            self.fs = f.fs
        elif self.fs != f.fs:
            mcep_data = f.resample_mel_cepstrum(self.fs).data

        return mcep_data[:, 1:]  # Drop 1st (power) dimension


class MelCepstrumFeatureConverter(abc.MapFeatureConverter):
    def __init__(self, base, mcep_fs=None):
        super().__init__(base)
        self.mcep_fs = mcep_fs

    def train(self, dataset, keys, **kwargs):
        mcep_dataset = MelCepstrumDataset(dataset, mcep_fs=self.mcep_fs)
        self.base.train(mcep_dataset, keys, **kwargs)
        self.order = mcep_dataset.order
        self.fs = mcep_dataset.fs

    def convert(self, mel_cepstrum, **kwargs):
        if self.order != mel_cepstrum.order:
            raise ValueError(f'order is expected to {self.order!s}'
                             f' but {mel_cepstrum.order!s}')

        if self.fs != mel_cepstrum.fs:
            result = kwiiyatta.resample(mel_cepstrum, self.fs)
        else:
            result = copy.copy(mel_cepstrum)

        result.data = np.hstack((
            result.data[:, 0].reshape(-1, 1),
            super().convert(result.data[:, 1:], raw=mel_cepstrum, **kwargs)))

        return result
