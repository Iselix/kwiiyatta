import pathlib
import subprocess

import pytest

import kwiiyatta

from tests import dataset, feature
from tests.plugin import assert_any


pytestmark = pytest.mark.slow


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
    subprocess.run(
        [
            'python', 'convert_voice.py',
            '--data-root', str(dataset.DATASET_ROOT),
            '--result-dir', str(result_root),
            '--converter-seed', '0',
        ], check=True)

    assert (result_root/'arctic_a0009.diff.wav').is_file()
    assert (result_root/'arctic_a0009.synth.wav').is_file()

    expected = make_expected_feature('arctic_a0009.wav')

    act_diff = kwiiyatta.analyze_wav(result_root/'arctic_a0009.diff.wav')
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, act_diff)
    check.round_equal(0.12, f0_diff)
    check.round_equal(0.69, spec_diff)
    check.round_equal(0.072, ape_diff)
    check.round_equal(0.26, mcep_diff)

    act_synth = kwiiyatta.analyze_wav(result_root/'arctic_a0009.synth.wav')
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, act_synth)
    check.round_equal(0.11, f0_diff)
    check.round_equal(0.77, spec_diff)
    check.round_equal(0.080, ape_diff)
    check.round_equal(0.25, mcep_diff)


@pytest.mark.assert_any
def test_voice_conversion_fullset(tmpdir):
    result_root = pathlib.Path(tmpdir)
    subprocess.run(
        [
            'python', 'convert_voice.py',
            '--data-root', str(dataset.get_dataset_path(dataset.DATASET_ROOT,
                                                        fullset=True)),
            '--result-dir', str(result_root),
            '--converter-seed', '0',
        ], check=True)

    results = ['arctic_a0036', 'arctic_a0041', 'arctic_a0082']

    for result in results:
        result_path = (result_root/result).with_suffix('.diff.wav')
        assert result_path.is_file()

        expected = make_expected_feature(result+'.wav', fullset=True)

        actual = kwiiyatta.analyze_wav(result_path)
        f0_diff, spec_diff, ape_diff, mcep_diff = \
            feature.calc_feature_diffs(expected, actual)
        assert_any.between(0.028, f0_diff, 0.054)
        assert_any.between(0.35, spec_diff, 0.42)
        assert_any.between(0.026, ape_diff, 0.044)
        assert_any.between(0.17, mcep_diff, 0.18)

    for result in results:
        result_path = (result_root/result).with_suffix('.synth.wav')
        assert result_path.is_file()

        expected = make_expected_feature(result+'.wav', fullset=True)

        actual = kwiiyatta.analyze_wav(result_path)
        f0_diff, spec_diff, ape_diff, mcep_diff = \
            feature.calc_feature_diffs(expected, actual)
        assert_any.between(0.048, f0_diff, 0.11)
        assert_any.between(0.36, spec_diff, 0.47)
        assert_any.between(0.068, ape_diff, 0.11)
        assert_any.between(0.17, mcep_diff, 0.19)
