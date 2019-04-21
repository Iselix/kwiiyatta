import functools

import numpy as np

import kwiiyatta


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
            calc_diff(exp.aperiodicity, act.aperiodicity, **kwargs),
            calc_diff(exp.mel_cepstrum.data, act.mel_cepstrum.data, **kwargs))


def override_spectrum_power(dest, tgt):
    dest.spectrum_envelope[:] *= \
        np.repeat(np.exp(np.mean(np.log(tgt.spectrum_envelope), axis=1)
                         - np.mean(np.log(dest.spectrum_envelope), axis=1))
                  .reshape(-1, 1),
                  dest.spectrum_len,
                  axis=1)


@functools.lru_cache(maxsize=None)
def _cached_analyzer(wavfile, frame_period):
    return kwiiyatta.analyze_wav(wavfile, frame_period=frame_period)


def get_analyzer(wavfile, frame_period=5, mcep_order=24):
    a = _cached_analyzer(wavfile, frame_period)
    a.mel_cepstrum_order = mcep_order
    return a
