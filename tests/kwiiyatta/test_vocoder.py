import copy
import functools

import pytest

import kwiiyatta

from tests import dataset, feature
from tests.plugin import assert_any


FRAME_PERIODS = [3, 5, 8]


@functools.lru_cache(maxsize=None)
def get_analyzer(wavfile, frame_period=5):
    return kwiiyatta.analyze_wav(wavfile, frame_period=frame_period)


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

    f0_diff, spec_diff, ape_diff = \
        feature.calc_feature_diffs(a1, a2, strict=False)
    check.round_equal(0.63, f0_diff)
    check.round_equal(1.0, spec_diff)
    check.round_equal(0.49, ape_diff)


@pytest.mark.assert_any
@pytest.mark.parametrize('wavfile', [dataset.CLB_WAV])
@pytest.mark.parametrize('frame_period', FRAME_PERIODS)
def test_reanalyze(wavfile, frame_period):
    a1 = get_analyzer(wavfile, frame_period=frame_period)

    analyzer_wav = a1.synthesize()
    feature_wav = kwiiyatta.feature(a1).synthesize()
    assert analyzer_wav.fs == feature_wav.fs
    assert (analyzer_wav.data == feature_wav.data).all()

    a2 = kwiiyatta.Analyzer(analyzer_wav,
                            frame_period=frame_period)

    f0_diff, spec_diff, ape_diff = feature.calc_feature_diffs(a1, a2)
    assert_any.between(0.079, f0_diff, 0.081)
    assert_any.between(0.20, spec_diff, 0.22)
    assert_any.between(0.070, ape_diff, 0.084)


def test_feature():
    a = get_analyzer(dataset.CLB_WAV)
    f = kwiiyatta.feature(a)

    assert f == a

    f._fs *= 2
    assert f != a
    f._fs = a.fs
    assert f == a

    f._frame_period *= 2
    assert f != a
    f._frame_period = a.frame_period
    assert f == a

    f.f0 = None
    assert f != a

    f.f0 = a.f0
    f.spectrum_envelope = copy.copy(a.spectrum_envelope)
    assert f == a

    f.spectrum_envelope[0][0] += 0.001
    assert f != a
    f.spectrum_envelope[0][0] = a.spectrum_envelope[0][0]
    f.aperiodicity = copy.copy(f.aperiodicity)
    assert f == a

    f.aperiodicity[-1][-1] += 0.001
    assert f != a
    f.aperiodicity[-1][-1] = a.aperiodicity[-1][-1]
    assert f == a
