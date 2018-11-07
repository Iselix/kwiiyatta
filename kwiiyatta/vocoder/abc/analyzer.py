import abc

import kwiiyatta

from .feature import Feature


class Analyzer(Feature):
    def __init__(self, wavdata, **kwargs):
        super().__init__(wavdata.fs, **kwargs)
        self._data = wavdata.data
        self._f0 = self._spectrum_envelope = self._aperiodicity = None

    @classmethod
    def load_wav(cls, wavfile, **kwargs):
        return cls(kwiiyatta.load_wav(wavfile), **kwargs)

    @property
    def data(self):
        return self._data

    @property
    def wavdata(self):
        return kwiiyatta.Wavdata(self.fs, self.data)

    @abc.abstractmethod
    def extract_f0(self):
        raise NotImplementedError

    def _get_f0(self):
        return self.extract_f0()

    @abc.abstractmethod
    def extract_spectrum_envelope(self):
        raise NotImplementedError

    def _get_spectrum_envelope(self):
        return self.extract_spectrum_envelope()

    @abc.abstractmethod
    def extract_aperiodicity(self):
        raise NotImplementedError

    def _get_aperiodicity(self):
        return self.extract_aperiodicity()
