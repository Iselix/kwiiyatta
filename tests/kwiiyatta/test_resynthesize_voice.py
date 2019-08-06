import pathlib
import sys

import pytest

import kwiiyatta
import kwiiyatta.resynthesize_voice as rv

from tests import dataset, feature


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    sys_argv = sys.argv
    yield
    sys.argv = sys_argv


def test_voice_resynthesis(tmpdir, check):
    result_root = pathlib.Path(tmpdir)
    sys.argv = [
        sys.argv[0],
        "--result-dir",
        str(result_root),
        str(dataset.CLB_WAV),
    ]
    rv.main()

    result_file = result_root/'arctic_a0001.wav'
    assert result_file.is_file()

    expected = kwiiyatta.analyze_wav(dataset.CLB_WAV)
    actual = kwiiyatta.analyze_wav(result_file)
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, actual)
    check.round_equal(0.079, f0_diff)
    check.round_equal(0.19, spec_diff)
    check.round_equal(0.073, ape_diff)
    check.round_equal(0.054, mcep_diff)


def test_voice_resynthesis_mcep(check, tmpdir):
    result_root = pathlib.Path(tmpdir)
    sys.argv = [
        sys.argv[0],
        "--result-dir", str(result_root),
        "--mcep",
        str(dataset.CLB_WAV),
    ]
    rv.main()

    result_file = result_root/'arctic_a0001.wav'
    assert result_file.is_file()

    expected = kwiiyatta.analyze_wav(dataset.CLB_WAV)
    actual = kwiiyatta.analyze_wav(result_file)
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, actual)
    check.round_equal(0.081, f0_diff)
    check.round_equal(0.22, spec_diff)
    check.round_equal(0.087, ape_diff)
    check.round_equal(0.051, mcep_diff)


def test_voice_resynthesis_carrier(check, tmpdir):
    result_root = pathlib.Path(tmpdir)
    sys.argv = [
        sys.argv[0],
        str(dataset.CLB_WAV),
        "--result-dir", str(result_root),
        "--mcep",
        "--mcep-order", "48",
        "--carrier", str(dataset.SLT_WAV)
    ]
    rv.main()

    result_file = result_root/'arctic_a0001.wav'
    assert result_file.is_file()

    clb = feature.get_analyzer(dataset.CLB_WAV)
    slt = feature.get_analyzer(dataset.SLT_WAV)
    expected = kwiiyatta.align(clb, slt)
    expected.f0 = slt.f0

    actual = kwiiyatta.analyze_wav(result_file)
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, actual)
    check.round_equal(0.065, f0_diff)
    check.round_equal(0.22, spec_diff)
    check.round_equal(0.083, ape_diff)
    check.round_equal(0.061, mcep_diff)


def test_voice_resynthesis_diffvc(check, tmpdir):
    result_root = pathlib.Path(tmpdir)
    sys.argv = [
        sys.argv[0],
        str(dataset.CLB_WAV),
        "--result-dir", str(result_root),
        "--mcep",
        "--mcep-order", "48",
        "--carrier", str(dataset.SLT_WAV),
        "--diffvc"
    ]
    rv.main()

    result_file = result_root/'arctic_a0001.wav'
    assert result_file.is_file()

    clb = feature.get_analyzer(dataset.CLB_WAV)
    slt = feature.get_analyzer(dataset.SLT_WAV)
    expected = kwiiyatta.align(clb, slt)
    expected.f0 = slt.f0
    feature.override_spectrum_power(expected, slt)
    expected.aperiodicity = slt.aperiodicity
    expected.mel_cepstrum = None

    actual = kwiiyatta.analyze_wav(result_file)
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, actual)
    check.round_equal(0.10, f0_diff)
    check.round_equal(0.35, spec_diff)
    check.round_equal(0.076, ape_diff)
    check.round_equal(0.079, mcep_diff)
