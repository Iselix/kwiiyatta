import itertools
import pathlib
import shutil
import sys

import numpy as np

import pytest

import kwiiyatta
import kwiiyatta.convert_voice as cv

from tests import dataset, feature
from tests.plugin import assert_any


pytestmark = pytest.mark.slow


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    sys_argv = sys.argv
    yield
    sys.argv = sys_argv


def make_expected_feature(wavpath, fs=16000, fullset=False):
    src = feature.get_analyzer(
        dataset.get_wav_path(dataset.CLB_DIR/wavpath, fullset, fs=fs))
    tgt = feature.get_analyzer(
        dataset.get_wav_path(dataset.SLT_DIR/wavpath, fullset, fs=fs))
    tgt_aligned = kwiiyatta.align(tgt, src)

    expected = kwiiyatta.feature(src)
    expected.spectrum_envelope = tgt_aligned.spectrum_envelope
    feature.override_spectrum_power(expected, src)

    return expected


def setup_dataset(src_dir):
    for num in range(1, 9):
        shutil.copy(dataset.CLB_DIR/f'arctic_a{num:04}.wav', src_dir)


def setup_44_dataset(src_dir):
    for num in range(1, 9):
        shutil.copy(dataset.get_wav_path(
                        dataset.CLB_DIR/f'arctic_a{num:04}.wav',
                        fs=44100),
                    src_dir)


def setup_dtype_dataset(src_dir):
    for num, dtype in zip(range(1, 9),
                          itertools.chain(dataset.DTYPES,
                                          itertools.repeat('i16'))):
        shutil.copy(
            dataset.get_wav_path(
                dataset.CLB_DIR/f'arctic_a{num:04}.wav',
                dtype=dtype,
            ), src_dir)


def setup_fs_dataset(src_dir):
    for num, fs in zip(range(1, 9),
                       itertools.chain(dataset.FS,
                                       itertools.repeat(16000))):
        shutil.copy(
            dataset.get_wav_path(
                dataset.CLB_DIR/f'arctic_a{num:04}.wav',
                fs=fs,
            ), src_dir)


@pytest.mark.assert_any
@pytest.mark.parametrize(
    'setup_func,target_fs,test_fs',
    [
        (setup_dataset,       16000, 16000),
        (setup_44_dataset,    44100, 44100),
        (setup_dataset,       16000, 44100),
        (setup_dtype_dataset, 16000, 16000),
        (setup_fs_dataset,    16000, 16000),
    ])
def test_voice_conversion(tmpdir, setup_func, target_fs, test_fs):
    tmp_path = pathlib.Path(tmpdir)
    result_root = tmp_path/'result'
    src_dir = tmp_path/'src'
    src_dir.mkdir(exist_ok=True)

    setup_func(src_dir)

    sys.argv = \
        [
            sys.argv[0],
            '--source', str(src_dir),
            '--target', str(dataset.get_dataset_path(dataset.SLT_DIR,
                                                     fs=target_fs)),
            '--result-dir', str(result_root),
            '--converter-seed', '0',
            '--converter-components', '1',
            '--max-files', '8',
            str(dataset.get_wav_path(dataset.CLB_DIR/'arctic_a0009.wav',
                                     fs=test_fs)),
        ]
    np.random.seed(0)
    cv.main()

    assert (result_root/'arctic_a0009.diff.wav').is_file()
    assert (result_root/'arctic_a0009.synth.wav').is_file()

    expected = make_expected_feature('arctic_a0009.wav', fs=test_fs)

    act_diff = kwiiyatta.analyze_wav(result_root/'arctic_a0009.diff.wav')
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, act_diff)
    assert_any.between(0.045, f0_diff, 0.12)
    assert_any.between(0.40, spec_diff, 0.48)
    assert_any.between(0.041, ape_diff, 0.055)
    assert_any.between(0.071, mcep_diff, 0.098)

    act_synth = kwiiyatta.analyze_wav(result_root/'arctic_a0009.synth.wav')
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, act_synth)
    assert_any.between(0.10, f0_diff, 0.12)
    assert_any.between(0.42, spec_diff, 0.50)
    assert_any.between(0.073, ape_diff, 0.094)
    assert_any.between(0.072, mcep_diff, 0.10)


@pytest.mark.assert_any
def test_voice_conversion_fullset(tmpdir):
    result_root = pathlib.Path(tmpdir)
    sys.argv = \
        [
            sys.argv[0],
            '--source',
            str(dataset.get_dataset_path(dataset.CLB_DIR, fullset=True)),
            '--target',
            str(dataset.get_dataset_path(dataset.SLT_DIR, fullset=True)),
            '--result-dir', str(result_root),
            '--converter-seed', '0',
            '--max-files', '100',
            '--skip-files', '3',
            str(dataset.get_wav_path(dataset.CLB_DIR/'arctic_a0001.wav',
                                     fullset=True)),
            str(dataset.get_wav_path(dataset.CLB_DIR/'arctic_a0002.wav',
                                     fullset=True)),
            str(dataset.get_wav_path(dataset.CLB_DIR/'arctic_a0003.wav',
                                     fullset=True)),
        ]
    np.random.seed(0)
    cv.main()

    results = ['arctic_a0001', 'arctic_a0002', 'arctic_a0003']

    for result in results:
        result_path = (result_root/result).with_suffix('.diff.wav')
        assert result_path.is_file()

        expected = make_expected_feature(result+'.wav', fullset=True)

        actual = kwiiyatta.analyze_wav(result_path)
        f0_diff, spec_diff, ape_diff, mcep_diff = \
            feature.calc_feature_diffs(expected, actual)
        assert_any.between(0.066, f0_diff, 0.10)
        assert_any.between(0.45, spec_diff, 0.47)
        assert_any.between(0.044, ape_diff, 0.059)
        assert_any.between(0.092, mcep_diff, 0.097)

    for result in results:
        result_path = (result_root/result).with_suffix('.synth.wav')
        assert result_path.is_file()

        expected = make_expected_feature(result+'.wav', fullset=True)

        actual = kwiiyatta.analyze_wav(result_path)
        f0_diff, spec_diff, ape_diff, mcep_diff = \
            feature.calc_feature_diffs(expected, actual)
        assert_any.between(0.060, f0_diff, 0.084)
        assert_any.between(0.44, spec_diff, 0.51)
        assert_any.between(0.10, ape_diff, 0.13)
        assert_any.between(0.095, mcep_diff, 0.098)
