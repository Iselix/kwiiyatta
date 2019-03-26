import copy
import itertools

from nnmnkwii.preprocessing.alignment import DTWAligner

import pytest

import kwiiyatta

from tests import dataset, feature
from tests.plugin import assert_any


FRAME_PERIODS = [3, 5, 8]
MCEP_ORDERS = [24, 36, 48]


def test_set_Analyzer_param():
    analyzer = kwiiyatta.analyze_wav(dataset.CLB_WAV)

    assert analyzer._f0 is None
    assert analyzer._spectrum_envelope is None
    assert analyzer._mel_cepstrum.data is None
    assert analyzer._aperiodicity is None

    _ = analyzer.aperiodicity
    assert analyzer._f0 is not None
    assert analyzer._spectrum_envelope is None
    assert analyzer._mel_cepstrum.data is None
    assert analyzer._aperiodicity is not None

    analyzer._aperiodicity = None
    _ = analyzer.mel_cepstrum
    assert analyzer._f0 is not None
    assert analyzer._spectrum_envelope is not None
    assert analyzer._mel_cepstrum.data is not None
    assert analyzer._aperiodicity is None

    feature = kwiiyatta.feature(analyzer)
    assert analyzer is not feature
    assert analyzer.mel_cepstrum_order == feature.mel_cepstrum_order
    assert analyzer._f0 is not None
    assert analyzer._f0 is feature.f0
    assert analyzer._spectrum_envelope is not None
    assert (analyzer._spectrum_envelope
            is feature._spectrum_envelope)
    assert analyzer.mel_cepstrum_order == feature.mel_cepstrum_order
    assert analyzer._mel_cepstrum.data is not None
    assert (analyzer._mel_cepstrum.data
            is feature._mel_cepstrum.data)
    assert analyzer._aperiodicity is not None
    assert analyzer._aperiodicity is feature.aperiodicity

    feature = feature[::2]
    f = copy.copy(feature)
    assert not feature.f0.flags['C_CONTIGUOUS']
    assert not feature.spectrum_envelope.flags['C_CONTIGUOUS']
    assert not feature.aperiodicity.flags['C_CONTIGUOUS']
    feature.ascontiguousarray()
    assert feature.f0.flags['C_CONTIGUOUS']
    assert feature.spectrum_envelope.flags['C_CONTIGUOUS']
    assert feature.aperiodicity.flags['C_CONTIGUOUS']
    assert feature is not f
    assert feature == f

    feature = kwiiyatta.feature(analyzer,
                                mcep_order=analyzer.mel_cepstrum_order*2)
    assert analyzer is not feature
    assert analyzer.mel_cepstrum_order != feature.mel_cepstrum_order
    assert analyzer._mel_cepstrum.data is not None
    assert (analyzer._mel_cepstrum.data
            is not feature.mel_cepstrum.data)
    assert ((analyzer._mel_cepstrum.data
             == feature.mel_cepstrum.data[:, :analyzer.mel_cepstrum_order+1])
            .all())

    feature.mel_cepstrum_order = analyzer.mel_cepstrum_order
    assert analyzer is not feature
    assert analyzer.mel_cepstrum_order == feature.mel_cepstrum_order
    assert analyzer._mel_cepstrum.data is not None
    assert (analyzer._mel_cepstrum.data
            is not feature.mel_cepstrum.data)
    assert (analyzer._mel_cepstrum.data
            == feature.mel_cepstrum.data).all()

    feature.f0 = None
    assert analyzer.f0 is not None

    feature.spectrum_envelope = None
    assert analyzer.spectrum_envelope is not None


@pytest.mark.parametrize('fs', dataset.FS)
def test_analyzer_feature(fs):
    from kwiiyatta.vocoder.world import WorldSynthesizer
    a = kwiiyatta.analyze_wav(dataset.get_wav_path(dataset.CLB_WAV, fs=fs),
                              frame_period=10, mcep_order=36)
    assert a.mel_cepstrum_order == 36
    assert a.spectrum_len == WorldSynthesizer.fs_spectrum_len(a.fs)

    f = kwiiyatta.feature(a)

    assert a is not f
    assert a.fs == f.fs
    assert a.frame_period == f.frame_period
    assert a.spectrum_len == f.spectrum_len
    assert a.mel_cepstrum_order == f.mel_cepstrum_order
    assert feature.calc_diff(a.f0, f.f0) == 0
    assert feature.calc_powered_diff(a.spectrum_envelope,
                                     f.spectrum_envelope) == 0
    assert feature.calc_diff(a.aperiodicity, f.aperiodicity) == 0
    assert f._mel_cepstrum.data is None
    assert feature.calc_diff(a.mel_cepstrum.data, f.mel_cepstrum.data) == 0


def test_analyze_difffile(check):
    a1 = feature.get_analyzer(dataset.CLB_WAV)
    a2 = feature.get_analyzer(dataset.CLB_WAV2)

    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(a1, a2, strict=False)
    check.round_equal(0.63, f0_diff)
    check.round_equal(1.0, spec_diff)
    check.round_equal(0.49, ape_diff)
    check.round_equal(0.47, mcep_diff)


@pytest.mark.assert_any
@pytest.mark.parametrize('wavfile', [dataset.CLB_WAV])
@pytest.mark.parametrize('dtype, fs',
                         itertools.chain(
                             itertools.product(dataset.DTYPES, [16000]),
                             itertools.product(
                                 ['i16'],
                                 (fs for fs in dataset.FS if fs != 16000)),
                         ))
@pytest.mark.parametrize('frame_period', FRAME_PERIODS)
def test_reanalyze(wavfile, dtype, fs, frame_period):
    a1 = feature.get_analyzer(
        dataset.get_wav_path(wavfile, dtype=dtype, fs=fs),
        frame_period=frame_period)
    assert a1.fs == fs

    analyzer_wav = a1.synthesize()
    feature_wav = kwiiyatta.feature(a1).synthesize()
    assert analyzer_wav.fs == feature_wav.fs
    assert (analyzer_wav.data == feature_wav.data).all()

    a2 = kwiiyatta.Analyzer(analyzer_wav,
                            frame_period=frame_period)

    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(a1, a2)
    assert_any.between(0.052, f0_diff, 0.094)
    assert_any.between(0.20, spec_diff, 0.22)
    assert_any.between(0.062, ape_diff, 0.097)
    assert_any.between(0.045, mcep_diff, 0.10)


def test_feature():
    a = feature.get_analyzer(dataset.CLB_WAV)
    f = kwiiyatta.feature(a)

    assert f == a

    f._mel_cepstrum._fs *= 2
    assert f != a
    f._mel_cepstrum._fs = a.fs
    assert f == a

    f._mel_cepstrum._frame_period *= 2
    assert f != a
    f._mel_cepstrum._frame_period = a.frame_period
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

    half = len(a.f0)//2
    f0, spec, ape, mcep = a[half]
    assert f0 == a.f0[half]
    assert (spec == a.spectrum_envelope[half]).all()
    assert (ape == a.aperiodicity[half]).all()
    assert (mcep == a.mel_cepstrum.data[half]).all()

    f = a[:half]
    assert len(f.f0) == len(f.spectrum_envelope) == len(f.aperiodicity) == half
    assert (f.f0 == a.f0[:half]).all()
    assert (f.spectrum_envelope == a.spectrum_envelope[:half]).all()
    assert (f.aperiodicity == a.aperiodicity[:half]).all()
    assert (f.mel_cepstrum.data == a.mel_cepstrum.data[:half]).all()


@pytest.mark.assert_any
@pytest.mark.parametrize('wavfile', [dataset.CLB_WAV])
@pytest.mark.parametrize('mcep_order', MCEP_ORDERS)
def test_mcep_to_spec(wavfile, mcep_order):
    a1 = kwiiyatta.analyze_wav(wavfile, mcep_order=mcep_order)
    mcep = a1.mel_cepstrum

    assert mcep.data.shape[-1] == mcep_order+1
    spec_diff = feature.calc_powered_diff(
        a1.spectrum_envelope,
        mcep.extract_spectrum(a1.spectrum_len))
    assert_any.between(0.021, spec_diff, 0.091)


def test_align_even():
    a1 = feature.get_analyzer(dataset.CLB_WAV)
    a2 = feature.get_analyzer(dataset.SLT_WAV)

    mcep1 = a1.mel_cepstrum.data
    mcep2 = a2.mel_cepstrum.data
    mcep1 = mcep1.reshape(1, *mcep1.shape)
    mcep2 = mcep2.reshape(1, *mcep2.shape)
    mcep1, mcep2 = DTWAligner(verbose=0).transform((mcep1, mcep2))
    exp_m1 = mcep1[0, :, :]
    exp_m2 = mcep2[0, :, :]

    act1, act2 = kwiiyatta.align_even(a1, a2)

    assert (exp_m1 == act1.mel_cepstrum.data).all()
    assert (exp_m2 == act2.mel_cepstrum.data).all()


@pytest.mark.assert_any
@pytest.mark.parametrize('fs1,fs2', dataset.FS_COMB)
@pytest.mark.parametrize('wavfile', [dataset.CLB_WAV])
@pytest.mark.parametrize('frame_period', FRAME_PERIODS)
def test_resample(fs1, fs2, wavfile, frame_period):
    if fs2 < fs1:
        fs1, fs2 = fs2, fs1
    a1 = feature.get_analyzer(dataset.get_wav_path(wavfile, fs=fs1),
                              frame_period=frame_period)
    a2 = feature.get_analyzer(dataset.get_wav_path(wavfile, fs=fs2),
                              frame_period=frame_period)

    f0_diff = feature.calc_diff(a1.f0, a2.f0)
    assert_any.between(0.0012, f0_diff, 0.014)
