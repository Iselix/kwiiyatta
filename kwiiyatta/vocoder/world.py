import numpy as np

import pyworld

import scipy

import kwiiyatta
from kwiiyatta.wavfile import Wavdata

from . import abc


class WorldAnalyzer(abc.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = np.ascontiguousarray(self._data)
        self._timeaxis = None

    @property
    def frame_len(self):
        return self.data.shape[0] * 1000 // self.fs // self.frame_period + 1

    @property
    def spectrum_len(self):
        if self._spectrum_envelope is not None:
            return self._spectrum_envelope.shape[-1]
        return WorldSynthesizer.fs_spectrum_len(self.fs)

    def clear_features(self):
        super().clear_features()
        self._f0 = self._spectrum_envelope = self._aperiodicity = None

    def extract_f0(self, **kwargs):
        if self._f0 is None:
            self._f0, self._timeaxis = pyworld.dio(
                self.data, self.fs, frame_period=self.frame_period,
                **kwargs
            )
            self._f0 = pyworld.stonemask(
                self.data, self._f0, self._timeaxis, self.fs)
        return self._f0

    def extract_spectrum_envelope(self, **kwargs):
        if self._spectrum_envelope is None:
            self._spectrum_envelope = pyworld.cheaptrick(
                self.data, self.f0, self._timeaxis, self.fs,
                **kwargs
            )
            # World のスペクトル包絡のパワーはサンプリングレートに比例する？
            self._spectrum_envelope /= self.fs
        return self._spectrum_envelope

    def extract_aperiodicity(self, **kwargs):
        if self._aperiodicity is None:
            self._aperiodicity = pyworld.d4c(
                self.data, self.f0, self._timeaxis, self.fs,
                **kwargs
            )
        return self._aperiodicity

    def ascontiguousarray(self):
        pass  # world で抽出した特徴量は既に C-contiguous


class WorldSynthesizer(abc.Synthesizer):
    SAFE_GUARD_MINIMUM = 0.000000000001
    EPS = 0.00000000000000022204460492503131
    FREQUENCY_INTERVAL = 3000.0
    UPPER_LIMIT = 15000.0

    @staticmethod
    def reshape_feature(feature):
        spectrum_len = feature.spectrum_len
        msb = spectrum_len.bit_length()
        reshape_spectrum_len = (1 << msb-1)
        if reshape_spectrum_len < spectrum_len-1:
            reshape_spectrum_len *= 2
        return kwiiyatta.reshape(feature, reshape_spectrum_len+1)

    @classmethod
    def synthesize(cls, feature):
        synth_feature = cls.reshape_feature(feature)
        synth_feature.ascontiguousarray()
        return Wavdata(
            synth_feature.fs,
            pyworld.synthesize(
                synth_feature.f0,
                synth_feature.spectrum_envelope * synth_feature.fs,
                synth_feature.aperiodicity,
                synth_feature.fs,
                synth_feature.frame_period
            ))

    @staticmethod
    def fs_spectrum_len(fs):
        return pyworld.get_cheaptrick_fft_size(fs) // 2 + 1

    @classmethod
    def _reshape_aperiodicity(cls, feature, fs, new_spectrum_len):
        feature = np.ascontiguousarray(feature)
        coded_ap = pyworld.code_aperiodicity(feature, fs)
        return pyworld.decode_aperiodicity(coded_ap, fs,
                                           (new_spectrum_len-1)*2)

    @classmethod
    def _resample_up_spectrum_envelope(cls, feature, fs,
                                       new_fs, new_spectrum_len):
        pad_spectrum_len = new_spectrum_len - feature.shape[1]
        return np.hstack((
            feature,
            np.abs(np.random.normal(
                0, cls.EPS/fs,
                (feature.shape[0], pad_spectrum_len)))
        ))

    @classmethod
    def _get_aperiodicity_num(cls, fs):
        return int(min(cls.UPPER_LIMIT, fs/2 - cls.FREQUENCY_INTERVAL)
                   / cls.FREQUENCY_INTERVAL)

    @classmethod
    def _resample_down_aperiodicity(cls, feature, fs,
                                    new_fs, new_spectrum_len):
        feature = np.ascontiguousarray(feature)
        coded_ap = pyworld.code_aperiodicity(feature, fs)
        num = cls._get_aperiodicity_num(new_fs)
        if num < coded_ap.shape[1]:
            coded_ap = np.ascontiguousarray(coded_ap[:, :num])
        return pyworld.decode_aperiodicity(coded_ap, new_fs,
                                           (new_spectrum_len-1)*2)

    @classmethod
    def _resample_up_aperiodicity(cls, feature, fs, new_fs, new_spectrum_len):
        feature = np.ascontiguousarray(feature)
        coded_ap = pyworld.code_aperiodicity(feature, fs)
        num = cls._get_aperiodicity_num(new_fs)
        if num > coded_ap.shape[1]:
            freq_axis = np.hstack((np.arange(coded_ap.shape[1]),
                                   new_fs / 2 / cls.FREQUENCY_INTERVAL - 1))
            coded_ap = np.hstack((coded_ap,
                                  np.full((coded_ap.shape[0], 1),
                                          -cls.SAFE_GUARD_MINIMUM)))
            ap_interp = scipy.interpolate.interp1d(freq_axis, coded_ap, axis=1)
            coded_ap = np.ascontiguousarray(ap_interp(np.arange(num)))
        return pyworld.decode_aperiodicity(coded_ap, new_fs,
                                           (new_spectrum_len-1)*2)
