from .dataset import (AlignedDataset, ParallelDataset, TrimmedDataset,
                      WavFileDataset,
                      make_dataset_to_array)
from .delta import DELTA_WINDOWS, DeltaFeatureConverter, DeltaFeatureDataset
from .gmm import GMMFeatureConverter
from .mcep import MelCepstrumDataset, MelCepstrumFeatureConverter


def MelCepstrumConverter(use_delta=True, Converter=GMMFeatureConverter,
                         **kwargs):
    converter = Converter(**kwargs)
    if use_delta:
        converter = DeltaFeatureConverter(converter)
    return MelCepstrumFeatureConverter(converter)


def align_dataset(parallel_dataset):
    return AlignedDataset(TrimmedDataset(parallel_dataset))


__all__ = ['MelCepstrumConverter', 'align_dataset']
__all__ += ['AlignedDataset', 'ParallelDataset', 'TrimmedDataset',
            'WavFileDataset',
            'make_dataset_to_array']
__all__ += ['GMMFeatureConverter']
__all__ += ['DELTA_WINDOWS', 'DeltaFeatureConverter', 'DeltaFeatureDataset']
__all__ += ['MelCepstrumDataset', 'MelCepstrumFeatureConverter']
