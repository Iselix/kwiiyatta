import copy

from nnmnkwii.preprocessing import remove_zeros_frames, trim_zeros_frames

import numpy as np

import kwiiyatta

from . import abc


class WavFileDataset(abc.Dataset):
    def __init__(self, data_dir, Analyzer=None):
        super().__init__()
        if Analyzer is None:
            Analyzer = kwiiyatta.analyze_wav
        self.Analyzer = Analyzer
        self.data_dir = data_dir
        if not self.data_dir.exists():
            raise FileNotFoundError(f'wav files dir "{self.data_dir!s}"'
                                    f' is not found')
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f'wav files dir "{self.data_dir!s}"'
                                     f' is not directory')
        self.files = frozenset(f.relative_to(self.data_dir) for f
                               in self.data_dir.glob('*.wav'))

    def keys(self):
        return self.files

    def get_data(self, key):
        return self.Analyzer(self.data_dir/key)


class ParallelDataset(abc.Dataset):
    def __init__(self, dataset1, dataset2):
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.common_keys = self.dataset1.keys() & self.dataset2.keys()

    def keys(self):
        return self.common_keys

    def get_data(self, key):
        return self.dataset1[key], self.dataset2[key]


@abc.map_dataset()
def TrimmedDataset(feature):
    s = trim_zeros_frames(feature.spectrum_envelope)
    return feature[:len(s)]  # トリムするフレームが手前にずれてるのでは？


@abc.map_dataset(expand_tuple=False)
def AlignedDataset(features):
    a, b = features
    return kwiiyatta.align_even(a, b)


def make_dataset_to_array(dataset, keys=None):
    if keys is None:
        keys = sorted(dataset.keys())

    data = None
    for key in keys:
        d = dataset[key]
        if isinstance(d, tuple):
            d = np.hstack(d)
        d = remove_zeros_frames(d)
        if data is None:
            data = copy.copy(d)
        else:
            len_data = len(data)
            data.resize(len_data+len(d), d.shape[-1])
            data[len_data:, :] = d
    return data
