import pathlib

import kwiiyatta

from tests import dataset


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
