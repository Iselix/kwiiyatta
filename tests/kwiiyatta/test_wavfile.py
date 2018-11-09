import pathlib

import pytest

import kwiiyatta

from tests import dataset
from tests.plugin import assert_any


@pytest.mark.assert_any
@pytest.mark.parametrize('wavfile',
                         [dataset.CLB_WAV, dataset.CLB_WAV2])
def test_load_and_save_wav(tmpdir, wavfile):
    wav = kwiiyatta.load_wav(wavfile)

    assert wav.fs == 16000
    assert -1 < wav.data.mean() < 1

    data_max = wav.data.max()
    assert_any.between(18616, data_max, 19970, sig_dig=5)

    data_min = wav.data.min()
    assert_any.between(-21297, data_min, -21297, sig_dig=5)

    savepath = pathlib.Path(tmpdir)/'save.wav'
    wav.save(savepath)

    saved_wav = kwiiyatta.load_wav(savepath)
    assert saved_wav.fs == wav.fs
    assert (saved_wav.data == wav.data).all()
