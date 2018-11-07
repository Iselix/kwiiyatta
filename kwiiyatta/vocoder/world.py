import pyworld

from kwiiyatta.wavfile import Wavdata

from . import abc


class WorldAnalyzer(abc.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeaxis = None

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


class WorldSynthesizer(abc.Synthesizer):
    @staticmethod
    def synthesize(feature):
        return Wavdata(
            feature.fs,
            pyworld.synthesize(
                feature.f0,
                feature.spectrum_envelope,
                feature.aperiodicity,
                feature.fs, feature.frame_period
            ))
