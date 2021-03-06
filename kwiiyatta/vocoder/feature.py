import copy

import numpy as np

import kwiiyatta

from . import abc


def feature(arg, **kwargs):
    if isinstance(arg, int):
        # arg is fs
        return Feature(arg, **kwargs)
    if isinstance(arg, abc.Feature):
        return Feature.init(arg, **kwargs)
    raise TypeError("argument should be int or Feature")


def pad_silence(feature, frame_len):
    padded_feature = kwiiyatta.feature(feature)
    padded_feature.f0 = np.concatenate((
        feature.Synthesizer.silence_f0(frame_len, feature.fs),
        feature.f0,
        feature.Synthesizer.silence_f0(frame_len, feature.fs)
    ))
    padded_feature.spectrum_envelope = np.concatenate((
        feature.Synthesizer.silence_spectrum_envelope(
            frame_len, feature.fs, feature.spectrum_len),
        feature.spectrum_envelope,
        feature.Synthesizer.silence_spectrum_envelope(
            frame_len, feature.fs, feature.spectrum_len)
    ))
    padded_feature.aperiodicity = np.concatenate((
        feature.Synthesizer.silence_aperiodicity(
            frame_len, feature.fs, feature.spectrum_len),
        feature.aperiodicity,
        feature.Synthesizer.silence_aperiodicity(
            frame_len, feature.fs, feature.spectrum_len)
    ))

    return padded_feature


class Feature(abc.MutableFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._f0 = self._spectrum_envelope = self._aperiodicity = None

    @classmethod
    def init(cls, feature, **kwargs):
        if 'frame_period' not in kwargs:
            kwargs['frame_period'] = feature.frame_period
        if 'mcep_order' not in kwargs:
            kwargs['mcep_order'] = feature.mel_cepstrum_order
        if 'Synthesizer' not in kwargs:
            kwargs['Synthesizer'] = feature.Synthesizer
        other = cls(feature.fs, **kwargs)
        other._f0 = feature.f0
        other._spectrum_envelope = feature.spectrum_envelope
        other._aperiodicity = feature.aperiodicity
        other._mel_cepstrum = copy.copy(feature._mel_cepstrum)
        return other

    @property
    def spectrum_len(self):
        if self._spectrum_envelope is not None:
            return self._spectrum_envelope.shape[-1]
        if self._aperiodicity is not None:
            return self._aperiodicity.shape[-1]
        return super().spectrum_len

    def _get_f0(self):
        return self._f0

    def _set_f0(self, value):
        self._f0 = value

    def _get_spectrum_envelope(self):
        return self._spectrum_envelope

    def _set_spectrum_envelope(self, value):
        self._spectrum_envelope = value

    def _get_aperiodicity(self):
        return self._aperiodicity

    def _set_aperiodicity(self, value):
        self._aperiodicity = value

    def synthesize(self):
        return self.Synthesizer.synthesize(self)
