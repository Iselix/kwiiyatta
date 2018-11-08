import numpy as np


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
