import abc

import kwiiyatta


class Analyzer(abc.ABC):
    def __init__(self, wavdata, *, frame_period=5):
        self._fs = wavdata.fs
        self._data = wavdata.data
        self._frame_period = frame_period
        self._f0 = self._spectrum_envelope = self._aperiodicity = None

    @classmethod
    def load_wav(cls, wavfile, **kwargs):
        return cls(kwiiyatta.load_wav(wavfile), **kwargs)

    @property
    def frame_period(self):
        return self._frame_period

    @property
    def fs(self):
        return self._fs

    @property
    def data(self):
        return self._data

    @property
    def wavdata(self):
        return kwiiyatta.Wavdata(self.fs, self.data)

    @abc.abstractmethod
    def extract_f0(self):
        raise NotImplementedError

    @property
    def f0(self):
        return self.extract_f0()

    @abc.abstractmethod
    def extract_spectrum_envelope(self):
        raise NotImplementedError

    @property
    def spectrum_envelope(self):
        return self.extract_spectrum_envelope()

    @abc.abstractmethod
    def extract_aperiodicity(self):
        raise NotImplementedError

    @property
    def aperiodicity(self):
        return self.extract_aperiodicity()
