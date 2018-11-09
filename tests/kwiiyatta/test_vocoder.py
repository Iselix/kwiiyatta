import functools

import numpy as np

import pytest

import pyworld

import kwiiyatta

from tests import dataset
from tests.plugin import assert_any


FRAME_PERIODS = [3, 5, 8]


@functools.lru_cache(maxsize=None)
def get_analyzer(wavfile, frame_period=5):
    return kwiiyatta.analyze_wav(wavfile, frame_period=frame_period)


def calc_diff(expected, actual, strict=True):
    assert not strict or abs(len(expected) - len(actual)) <= 1
    if len(expected) < len(actual):
        actual = actual[:len(expected)]
    elif len(expected) > len(actual):
        expected = expected[:len(actual)]
    norm = (np.abs if len(expected.shape) == 1
            else lambda x: np.linalg.norm(x, axis=1))
    return np.mean(norm(expected - actual)) / np.mean(norm(expected))


def calc_powered_diff(expected, actual, **kwargs):
    return calc_diff(np.sqrt(expected), np.sqrt(actual), **kwargs)


def calc_feature_diffs(exp, act, **kwargs):
    return (calc_diff(exp.f0, act.f0, **kwargs),
            calc_powered_diff(exp.spectrum_envelope,
                              act.spectrum_envelope,
                              **kwargs),
            calc_diff(exp.aperiodicity, act.aperiodicity, **kwargs))


def test_set_Analyzer_param():
    analyzer = kwiiyatta.analyze_wav(dataset.CLB_WAV)

    assert analyzer._f0 is None
    assert analyzer._spectrum_envelope is None
    assert analyzer._aperiodicity is None

    _ = analyzer.aperiodicity
    assert analyzer._f0 is not None
    assert analyzer._spectrum_envelope is None
    assert analyzer._aperiodicity is not None

    analyzer._aperiodicity = None
    _ = analyzer.spectrum_envelope
    assert analyzer._f0 is not None
    assert analyzer._spectrum_envelope is not None
    assert analyzer._aperiodicity is None


def test_analyze_difffile(check):
    a1 = get_analyzer(dataset.CLB_WAV)
    a2 = get_analyzer(dataset.CLB_WAV2)

    f0_diff, spec_diff, ape_diff = calc_feature_diffs(a1, a2, strict=False)
    check.round_equal(0.63, f0_diff)
    check.round_equal(1.0, spec_diff)
    check.round_equal(0.49, ape_diff)


@pytest.mark.assert_any
@pytest.mark.parametrize('wavfile', [dataset.CLB_WAV])
@pytest.mark.parametrize('frame_period', FRAME_PERIODS)
def test_reanalyze(wavfile, frame_period):
    a1 = get_analyzer(wavfile, frame_period=frame_period)

    waveform = pyworld.synthesize(a1.f0, a1.spectrum_envelope, a1.aperiodicity,
                                  a1.fs, a1.frame_period)

    a2 = kwiiyatta.Analyzer(kwiiyatta.Wavdata(a1.fs, waveform),
                            frame_period=frame_period)

    f0_diff, spec_diff, ape_diff = calc_feature_diffs(a1, a2)
    assert_any.between(0.079, f0_diff, 0.081)
    assert_any.between(0.20, spec_diff, 0.22)
    assert_any.between(0.070, ape_diff, 0.084)
