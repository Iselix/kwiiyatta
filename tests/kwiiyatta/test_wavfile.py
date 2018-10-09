import itertools
import pathlib

import numpy as np

import pytest

import kwiiyatta

from tests import dataset
from tests.plugin import assert_any


@pytest.mark.assert_any
@pytest.mark.parametrize('wavfile', [dataset.CLB_WAV, dataset.CLB_WAV2])
@pytest.mark.parametrize('dtype,fs', itertools.chain(
    itertools.product(dataset.DTYPES, [16000]),
    itertools.product(['i16'], dataset.FS)))
def test_load_wav(tmpdir, wavfile, dtype, fs):
    wav = kwiiyatta.load_wav(dataset.get_wav_path(wavfile, dtype=dtype, fs=fs))

    assert wav.fs == fs
    assert -0.005 < wav.data.mean() < 0.005

    data_max = wav.data.max()
    assert_any.between(0.56, data_max, 0.61)

    data_min = wav.data.min()
    assert_any.between(-0.67, data_min, -0.64)

    savepath = pathlib.Path(tmpdir)/'save.wav'
    wav.save(savepath, normalize=False)

    saved_wav = kwiiyatta.load_wav(savepath)
    assert saved_wav.fs == wav.fs
    assert np.abs(saved_wav.data - wav.data).max() == 0


@pytest.mark.assert_any
@pytest.mark.parametrize('wavfile',
                         [dataset.CLB_WAV, dataset.CLB_WAV2])
def test_load_and_save_wav_normalized(tmpdir, wavfile):
    wav = kwiiyatta.load_wav(wavfile)

    savepath = pathlib.Path(tmpdir)/'save.wav'
    wav.save(savepath, peak_lv=-1.5)

    expected_peak = np.power(10, -1.5/10) * 2**15

    saved_wav = kwiiyatta.load_wav(savepath)
    assert saved_wav.fs == wav.fs
    assert saved_wav.data.mean() < 1e-5
    assert np.abs(saved_wav.data).max() <= expected_peak

    data_dc = wav.data.mean()
    data_max = np.abs(wav.data - data_dc).max()
    expected_data = wav.data - data_dc
    if np.abs(expected_data).max() > expected_peak:
        expected_data *= expected_peak / data_max

    assert np.abs(saved_wav.data - expected_data).max() < 1e-4
