import pathlib
import sys

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


def make_expected_feature(wavpath, fullset=False):
    src = feature.get_analyzer(
        dataset.get_wav_path(dataset.CLB_DIR/wavpath, fullset))
    tgt = feature.get_analyzer(
        dataset.get_wav_path(dataset.SLT_DIR/wavpath, fullset))
    tgt_aligned = kwiiyatta.align(tgt, src)

    expected = kwiiyatta.feature(src)
    expected.spectrum_envelope = tgt_aligned.spectrum_envelope
    feature.override_spectrum_power(expected, src)

    return expected


def test_voice_conversion(tmpdir, check):
    result_root = pathlib.Path(tmpdir)
    sys.argv = \
        [
            sys.argv[0],
            '--source', str(dataset.CLB_DIR),
            '--target', str(dataset.SLT_DIR),
            '--result-dir', str(result_root),
            '--converter-seed', '0',
            '--converter-components', '1',
            '--max-files', '8',
            str(dataset.CLB_DIR/'arctic_a0009.wav'),
        ]
    cv.main()

    assert (result_root/'arctic_a0009.diff.wav').is_file()
    assert (result_root/'arctic_a0009.synth.wav').is_file()

    expected = make_expected_feature('arctic_a0009.wav')

    act_diff = kwiiyatta.analyze_wav(result_root/'arctic_a0009.diff.wav')
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, act_diff)
    check.round_equal(0.094, f0_diff)
    check.round_equal(0.44, spec_diff)
    check.round_equal(0.047, ape_diff)
    check.round_equal(0.16, mcep_diff)

    act_synth = kwiiyatta.analyze_wav(result_root/'arctic_a0009.synth.wav')
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, act_synth)
    check.round_equal(0.10, f0_diff)
    check.round_equal(0.49, spec_diff)
    check.round_equal(0.089, ape_diff)
    check.round_equal(0.16, mcep_diff)


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
    cv.main()

    results = ['arctic_a0001', 'arctic_a0002', 'arctic_a0003']

    for result in results:
        result_path = (result_root/result).with_suffix('.diff.wav')
        assert result_path.is_file()

        expected = make_expected_feature(result+'.wav', fullset=True)

        actual = kwiiyatta.analyze_wav(result_path)
        f0_diff, spec_diff, ape_diff, mcep_diff = \
            feature.calc_feature_diffs(expected, actual)
        assert_any.between(0.048, f0_diff, 0.10)
        assert_any.between(0.44, spec_diff, 0.47)
        assert_any.between(0.040, ape_diff, 0.068)
        assert_any.between(0.16, mcep_diff, 0.17)

    for result in results:
        result_path = (result_root/result).with_suffix('.synth.wav')
        assert result_path.is_file()

        expected = make_expected_feature(result+'.wav', fullset=True)

        actual = kwiiyatta.analyze_wav(result_path)
        f0_diff, spec_diff, ape_diff, mcep_diff = \
            feature.calc_feature_diffs(expected, actual)
        assert_any.between(0.064, f0_diff, 0.084)
        assert_any.between(0.44, spec_diff, 0.50)
        assert_any.between(0.099, ape_diff, 0.13)
        assert_any.between(0.16, mcep_diff, 0.18)
