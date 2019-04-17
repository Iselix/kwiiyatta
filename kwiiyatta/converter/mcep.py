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
