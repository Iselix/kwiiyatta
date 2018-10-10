import copy
import functools
import math

import numpy as np

import scipy.signal as ssig

import kwiiyatta


def reshape(feature, new_shape):
    shape = feature.shape[1]
    shape_gcd = math.gcd(shape, new_shape)
    pad_shape = shape//shape_gcd*20
    trim_shape = new_shape//shape_gcd*20

    feature = np.hstack((feature[:, 0][:, None].repeat(pad_shape, axis=1),
                         feature,
                         feature[:, -1][:, None].repeat(pad_shape, axis=1)))

    feature = ssig.resample_poly(feature, new_shape, shape, axis=1)
    return feature[:, trim_shape:-trim_shape]


def calc_diff(expected, actual, strict=True):
    assert not strict or abs(len(expected) - len(actual)) <= 1
    if len(expected) < len(actual):
        actual = actual[:len(expected)]
    elif len(expected) > len(actual):
        expected = expected[:len(actual)]

    if len(expected.shape) > 1:
        def norm(x):
            return np.linalg.norm(x, axis=1)

        exp_shape = expected.shape[1]
        act_shape = actual.shape[1]

        if exp_shape < act_shape:
            expected = reshape(expected, act_shape)
        elif act_shape < exp_shape:
            actual = reshape(actual, exp_shape)
    else:
        norm = np.abs

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


def override_power(dest, tgt, out=None):
    if out is None:
        out = copy.copy(dest)
    elif out is not dest:
        out[:] = dest
    out[:] *= np.repeat(np.exp(np.mean(np.log(tgt), axis=1)
                               - np.mean(np.log(dest), axis=1))
                        .reshape(-1, 1),
                        dest.shape[-1],
                        axis=1)
    return out


def override_spectrum_power(dest, tgt):
    override_power(dest.spectrum_envelope,
                   tgt.spectrum_envelope,
                   dest.spectrum_envelope)


@functools.lru_cache(maxsize=None)
def _cached_analyzer(wavfile, frame_period):
    return kwiiyatta.analyze_wav(wavfile, frame_period=frame_period)


def get_analyzer(wavfile, frame_period=5, mcep_order=24):
    a = _cached_analyzer(wavfile, frame_period)
    a.mel_cepstrum_order = mcep_order
    return a
