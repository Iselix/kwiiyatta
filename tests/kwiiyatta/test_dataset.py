import itertools
import pathlib

import pytest

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
