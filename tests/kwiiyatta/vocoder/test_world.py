import pytest

import pyworld

import kwiiyatta

import tests.dataset as dataset


def test_world_array_order():
    wav = kwiiyatta.load_wav(dataset.CLB_WAV)

    f0, timeaxis = pyworld.dio(wav.data, wav.fs)
    f0 = pyworld.stonemask(wav.data, f0, timeaxis, wav.fs)
    spec = pyworld.cheaptrick(wav.data, f0, timeaxis, wav.fs)
    ape = pyworld.d4c(wav.data, f0, timeaxis, wav.fs)
    pyworld.synthesize(f0, spec, ape, wav.fs)

    data = wav.data[::2]

    expected_msg = 'ndarray is not C-contiguous'
    with pytest.raises(ValueError) as e:
        f0, timeaxis = pyworld.dio(data, wav.fs)
    assert expected_msg == str(e.value)

    with pytest.raises(ValueError) as e:
        f0 = pyworld.stonemask(data, f0, timeaxis, wav.fs)
    assert expected_msg == str(e.value)

    with pytest.raises(ValueError) as e:
        pyworld.cheaptrick(data, f0, timeaxis, wav.fs)
    assert expected_msg == str(e.value)

    with pytest.raises(ValueError) as e:
        pyworld.d4c(data, f0, timeaxis, wav.fs)
    assert expected_msg == str(e.value)

    with pytest.raises(ValueError) as e:
        pyworld.synthesize(f0[::2], spec[::2], ape[::2], wav.fs)
    assert expected_msg == str(e.value)
