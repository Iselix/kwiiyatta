import abc

import numpy as np

import scipy

import kwiiyatta


class Synthesizer(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def synthesize(feature):
        raise NotImplementedError

    @staticmethod
    def fs_spectrum_len(fs):
        raise NotImplementedError

    @staticmethod
    def _reshape_feature(feature, fs, new_spectrum_len):
        spectrum_len = feature.shape[1]
        spectrum_len_gcd = np.gcd(spectrum_len, new_spectrum_len)
        pad_len = spectrum_len//spectrum_len_gcd*20
        trim_len = new_spectrum_len//spectrum_len_gcd*20

        feature = np.hstack((feature[:, 0][:, None].repeat(pad_len, axis=1),
                             feature,
                             feature[:, -1][:, None].repeat(pad_len, axis=1)))

        feature = scipy.signal.resample_poly(feature, new_spectrum_len,
                                             spectrum_len, axis=1)
        return feature[:, trim_len:-trim_len]

    @classmethod
    def reshape_spectrum_envelope(cls, feature, fs, new_spectrum_len):
        return np.exp(cls._reshape_feature(
            np.log(feature), fs, new_spectrum_len))

    @classmethod
    def reshape_aperiodicity(cls, feature, fs, new_spectrum_len):
        return np.exp(cls._reshape_feature(
            np.log(feature), fs, new_spectrum_len))

    @staticmethod
    def _resample_spectrum_len(feature, fs, new_fs):
        return feature.shape[1] * new_fs // fs

    @staticmethod
    def _resample_up(feature, fs, new_fs, new_spectrum_len, pad, window=1):
        spectrum_len = feature.shape[1]
        pad_spectrum_len = new_spectrum_len - spectrum_len
        feature = np.hstack((
            feature,
            pad[:, -pad_spectrum_len:]
        ))
        overlap = pad.shape[1] - pad_spectrum_len
        if overlap > 0:
            overlap_slice = slice(-pad_spectrum_len-overlap,
                                  -pad_spectrum_len)
            feature[:, overlap_slice] *= 1 - window
            feature[:, overlap_slice] += window * pad[:, :overlap]
        return feature

    @staticmethod
    def _resample_down(feature, fs, new_fs, new_spectrum_len):
        return feature[:, :new_spectrum_len]

    @staticmethod
    @abc.abstractmethod
    def _resample_up_spectrum_envelope(feature, fs, new_fs, new_spectrum_len):
        raise NotImplementedError

    _resample_down_spectrum_envelope = _resample_down

    @staticmethod
    @abc.abstractmethod
    def _resample_up_aperiodicity(feature, fs, new_fs, new_spectrum_len):
        raise NotImplementedError

    _resample_down_aperiodicity = _resample_down

    @classmethod
    def _resample(cls, feature, fs, new_fs, up_func, down_func):
        if fs == new_fs:
            return feature
        new_spectrum_len = cls._resample_spectrum_len(feature, fs, new_fs)
        if fs < new_fs:
            return up_func(feature, fs, new_fs, new_spectrum_len)
        return down_func(feature, fs, new_fs, new_spectrum_len)

    @classmethod
    def resample_spectrum_envelope(cls, feature, fs, new_fs):
        return cls._resample(feature, fs, new_fs,
                             cls._resample_up_spectrum_envelope,
                             cls._resample_down_spectrum_envelope)

    @classmethod
    def resample_aperiodicity(cls, feature, fs, new_fs):
        return cls._resample(feature, fs, new_fs,
                             cls._resample_up_aperiodicity,
                             cls._resample_down_aperiodicity)

    @staticmethod
    @abc.abstractmethod
    def extract_is_voiced(feature):
        raise NotImplementedError

    @classmethod
    def create_silence_feature(cls, frame_len, fs, **kwargs):
        kwargs.setdefault('Synthesizer', cls)
        feature = kwiiyatta.feature(fs, **kwargs)

        feature.f0 = cls.silence_f0(frame_len, fs)
        feature.spectrum_envelope = \
            cls.silence_spectrum_envelope(frame_len, fs)
        feature.aperiodicity = cls.silence_aperiodicity(frame_len, fs)

        return feature

    @staticmethod
    @abc.abstractmethod
    def silence_f0(frame_len, fs):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _silence_spectrum_envelope(frame_len, fs, spectrum_len):
        raise NotImplementedError

    @classmethod
    def silence_spectrum_envelope(cls, frame_len, fs, spectrum_len=None):
        if spectrum_len is None:
            spectrum_len = cls.fs_spectrum_len(fs)
        return cls._silence_spectrum_envelope(frame_len, fs, spectrum_len)

    @staticmethod
    @abc.abstractmethod
    def _silence_aperiodicity(frame_len, fs, spectrum_len):
        raise NotImplementedError

    @classmethod
    def silence_aperiodicity(cls, frame_len, fs, spectrum_len=None):
        if spectrum_len is None:
            spectrum_len = cls.fs_spectrum_len(fs)
        return cls._silence_aperiodicity(frame_len, fs, spectrum_len)

    @staticmethod
    def silence_is_voiced(frame_len, fs):
        return np.full((frame_len), False)
