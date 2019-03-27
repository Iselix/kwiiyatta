import abc

import numpy as np

import scipy


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
