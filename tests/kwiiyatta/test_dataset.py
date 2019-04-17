import copy
import itertools
import pathlib

from nnmnkwii.preprocessing import delta_features

import numpy as np

import pytest

from sklearn.model_selection import train_test_split

import kwiiyatta
from kwiiyatta.converter import (AlignedDataset, DeltaFeatureDataset,
                                 MelCepstrumDataset, TrimmedDataset,
                                 make_dataset_to_array)
from kwiiyatta.converter.delta import DELTA_WINDOWS

from tests import dataset, feature


def test_dataset():
    clb_dataset = kwiiyatta.WavFileDataset(dataset.CLB_DIR)
    expected_keys = {pathlib.Path(f'arctic_a{num:04}.wav') for num
                     in range(1, 10)}
    assert expected_keys == clb_dataset.keys()

    for key in expected_keys:
        expected = kwiiyatta.analyze_wav(dataset.CLB_DIR/key)
        actual = clb_dataset[key]
        assert (expected.f0 == actual.f0).all()
        assert (expected.spectrum_envelope == actual.spectrum_envelope).all()
        assert (expected.aperiodicity == actual.aperiodicity).all()
        assert (expected.mel_cepstrum.data == actual.mel_cepstrum.data).all()

    mcep_dataset = MelCepstrumDataset(clb_dataset)
    expected = kwiiyatta.analyze_wav(dataset.CLB_WAV).mel_cepstrum.data[:, 1:]
    assert (expected == mcep_dataset['arctic_a0001.wav']).all()


@pytest.mark.parametrize('fullset_clb, fullset_slt, fullset_expected',
                         [
                             (False, False, False),
                             (False, True,  False),
                             (True,  True,  True),
                         ])
def test_parallel_dataset(fullset_clb, fullset_slt, fullset_expected):
    clb_dataset = kwiiyatta.WavFileDataset(dataset.get_dataset_path(
        dataset.CLB_DIR, fullset=fullset_clb))
    slt_dataset = kwiiyatta.WavFileDataset(dataset.get_dataset_path(
        dataset.SLT_DIR, fullset=fullset_slt))
    parallel_dataset = kwiiyatta.ParallelDataset(clb_dataset, slt_dataset)

    if fullset_expected:
        assert (parallel_dataset.keys()
                == {pathlib.Path(f'arctic_{c}{num:04}.wav') for c, num
                    in itertools.chain(
                        (('a', num) for num in range(1, 594)),
                        (('b', num) for num in range(1, 540)),
                    )})
    else:
        assert (parallel_dataset.keys()
                == {pathlib.Path(f'arctic_a{num:04}.wav') for num
                    in range(1, 10)})

    clb_data = clb_dataset['arctic_a0001.wav']
    slt_data = slt_dataset['arctic_a0001.wav']
    p_clb_data, p_slt_data = parallel_dataset['arctic_a0001.wav']
    assert clb_data == p_clb_data
    assert slt_data == p_slt_data

    mcep_dataset = MelCepstrumDataset(parallel_dataset)
    m_clb_data, m_slt_data = mcep_dataset['arctic_a0001.wav']
    assert (clb_data.mel_cepstrum.data[:, 1:] == m_clb_data).all()
    assert (slt_data.mel_cepstrum.data[:, 1:] == m_slt_data).all()


@pytest.mark.xfail(strict=True,
                   reason='Trim position of TrimmedDataset is shifted')
def test_trimmed_dataset():
    def add_margin(data, margin_len=64):
        if len(data.shape) == 1:
            pad = np.zeros((margin_len,))
        else:
            pad = np.zeros((margin_len, data.shape[1]))
        return np.concatenate((pad, data, pad), axis=0)

    f = kwiiyatta.feature(feature.get_analyzer(dataset.CLB_WAV))
    f.f0 = add_margin(f.f0)
    f.spectrum_envelope = add_margin(f.spectrum_envelope)
    f.aperiodicity = add_margin(f.aperiodicity)
    len_f = len(f.f0)

    d = TrimmedDataset({'f': f})
    len_d = len(d['f'].f0)

    assert len_d == len_f-64
    assert np.abs(d['f'].spectrum_envelope[0]).sum() > 0
    assert np.abs(d['f'].spectrum_envelope[-1]).sum() > 0


def test_mcep_dataset():
    a = kwiiyatta.feature(feature.get_analyzer(dataset.CLB_WAV))
    base = {'order24': copy.copy(a)}
    base['order32'] = copy.copy(a)
    base['order32'].mel_cepstrum_order = 32
    base['fs44'] = copy.copy(a)
    base['fs44'].mel_cepstrum._fs = 44100

    mcep_dataset = MelCepstrumDataset(base)

    mcep_dataset['order24']
    with pytest.raises(ValueError) as e:
        mcep_dataset['order32']

    assert 'order of "order32" is 32 but others are 24' == str(e.value)

    with pytest.raises(ValueError) as e:
        mcep_dataset['fs44']

    assert 'fs of "fs44" is 44100 but others are 16000' == str(e.value)


def test_delta_dataset():
    base = {'fp5': feature.get_analyzer(dataset.CLB_WAV),
            'fp5_2': feature.get_analyzer(dataset.CLB_WAV2),
            'fp3': feature.get_analyzer(dataset.CLB_WAV, frame_period=3)}

    mcep_dataset = MelCepstrumDataset(base)
    delta_dataset = DeltaFeatureDataset(mcep_dataset)

    assert (delta_dataset['fp5']
            == delta_features(mcep_dataset['fp5'], DELTA_WINDOWS)).all()

    assert (delta_dataset['fp5_2']
            == delta_features(mcep_dataset['fp5_2'], DELTA_WINDOWS)).all()

    with pytest.raises(ValueError) as e:
        delta_dataset['fp3']

    assert 'frame_period of "fp3" is 3 but others are 5' == str(e.value)


def make_expected_dataset(data_root, use_delta):
    from pathlib import Path
    from nnmnkwii.datasets import PaddedFileSourceDataset
    from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
    from nnmnkwii.metrics import melcd
    from nnmnkwii.preprocessing import (delta_features, remove_zeros_frames,
                                        trim_zeros_frames)
    from nnmnkwii.preprocessing.alignment import DTWAligner
    from nnmnkwii.util import apply_each2d_trim

    max_files = 100  # number of utterances to be used.
    test_size = 0.03

    windows = DELTA_WINDOWS

    class MyFileDataSource(CMUArcticWavFileDataSource):
        def __init__(self, *args, **kwargs):
            super(MyFileDataSource, self).__init__(*args, **kwargs)
            self.test_paths = None

        def collect_files(self):
            paths = [Path(path) for path in super(
                MyFileDataSource, self).collect_files()]
            paths_train, paths_test = train_test_split(
                paths, test_size=test_size, random_state=1234)

            # keep paths for later testing
            self.test_paths = paths_test

            return paths_train

        def collect_features(self, path):
            feature = kwiiyatta.analyze_wav(path)
            s = trim_zeros_frames(feature.spectrum_envelope)
            return feature.mel_cepstrum.data[:len(s)]  # トリムするフレームが手前にずれてるのでは？

    clb_source = MyFileDataSource(data_root=data_root,
                                  speakers=["clb"], max_files=max_files)
    slt_source = MyFileDataSource(data_root=data_root,
                                  speakers=["slt"], max_files=max_files)

    X = PaddedFileSourceDataset(clb_source, 1200).asarray()
    Y = PaddedFileSourceDataset(slt_source, 1200).asarray()

    # Alignment
    X_aligned, Y_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y))

    # Drop 1st (power) dimension
    X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]

    if use_delta:
        X_aligned = apply_each2d_trim(delta_features, X_aligned, windows)
        Y_aligned = apply_each2d_trim(delta_features, Y_aligned, windows)

    XY = (np.concatenate((X_aligned, Y_aligned), axis=-1)
          .reshape(-1, X_aligned.shape[-1]*2))

    return remove_zeros_frames(XY)


@pytest.mark.parametrize('fullset_clb, fullset_slt, fullset_expected',
                         [
                             (False, False, False),
                             (False, True,  False),
                             pytest.param(True, True, True,
                                          marks=pytest.mark.slow),
                         ])
@pytest.mark.parametrize('use_delta', [False, True])
def test_dataset_array(fullset_clb, fullset_slt, fullset_expected, use_delta):
    clb = dataset.get_dataset_path(dataset.CLB_DIR, fullset_clb)
    slt = dataset.get_dataset_path(dataset.SLT_DIR, fullset_slt)

    d1 = kwiiyatta.WavFileDataset(clb)
    d2 = kwiiyatta.WavFileDataset(slt)
    parallel_dataset = \
        MelCepstrumDataset(
            AlignedDataset(
                TrimmedDataset(
                    kwiiyatta.ParallelDataset(d1, d2))))
    if use_delta:
        parallel_dataset = DeltaFeatureDataset(parallel_dataset)

    keys, _ = train_test_split(sorted(parallel_dataset.keys())[:100],
                               test_size=0.03, random_state=1234)

    expected = make_expected_dataset(
        dataset.FULLSET_ROOT if fullset_expected else dataset.DATASET_ROOT,
        use_delta
    )
    actual = make_dataset_to_array(parallel_dataset, keys)
    assert np.abs(expected - actual).max() < 1e-6
