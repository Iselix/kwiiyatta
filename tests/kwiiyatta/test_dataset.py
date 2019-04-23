import itertools
import pathlib

import numpy as np

import pytest

import kwiiyatta
from kwiiyatta.converter import TrimmedDataset

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
