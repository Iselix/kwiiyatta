import copy
import itertools
import pathlib

from nnmnkwii.preprocessing import delta_features

import numpy as np

import pytest

import kwiiyatta
from kwiiyatta.converter import (DeltaFeatureDataset, MelCepstrumDataset,
                                 TrimmedDataset)
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
