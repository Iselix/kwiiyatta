import numpy as np

import pyworld

from kwiiyatta.wavfile import Wavdata

from . import abc


class WorldAnalyzer(abc.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = np.ascontiguousarray(self._data)
        self._timeaxis = None

    @property
    def spectrum_len(self):
        if self._spectrum_envelope is not None:
            return self._spectrum_envelope.shape[-1]
        return WorldSynthesizer.fs_spectrum_len(self.fs)

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
    @staticmethod
    def synthesize(feature):
        feature.ascontiguousarray()
        return Wavdata(
            feature.fs,
            pyworld.synthesize(
                feature.f0,
                feature.spectrum_envelope,
                feature.aperiodicity,
                feature.fs, feature.frame_period
            ))

    @staticmethod
    def fs_spectrum_len(fs):
        return pyworld.get_cheaptrick_fft_size(fs) // 2 + 1
