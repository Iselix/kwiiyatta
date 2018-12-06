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
    check.round_equal(0.20, spec_diff)
    check.round_equal(0.073, ape_diff)
    check.round_equal(0.099, mcep_diff)


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
    check.round_equal(0.094, mcep_diff)
