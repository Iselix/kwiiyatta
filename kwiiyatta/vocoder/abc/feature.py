import abc


class Feature(abc.ABC):
    def __init__(self, fs, frame_period=5):
        self._fs = fs
        self._frame_period = frame_period

    @property
    def fs(self):
        return self._fs

    @property
    def frame_period(self):
        return self._frame_period

    @abc.abstractmethod
    def _get_f0(self):
        raise NotImplementedError

    @property
    def f0(self):
        return self._get_f0()

    @abc.abstractmethod
    def _get_spectrum_envelope(self):
        raise NotImplementedError

    @property
    def spectrum_envelope(self):
        return self._get_spectrum_envelope()

    @abc.abstractmethod
    def _get_aperiodicity(self):
        raise NotImplementedError

    @property
    def aperiodicity(self):
        return self._get_aperiodicity()

    def __eq__(self, other):
        if self.frame_period != other.frame_period or self.fs != other.fs:
            return False
        if ((self._get_f0() != other._get_f0()).any()
                or (self._get_spectrum_envelope()
                    != other._get_spectrum_envelope()).any()
                or (self._get_aperiodicity()
                    != other._get_aperiodicity()).any()):
            return False
        return True


class MutableFeature(Feature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def _set_f0(self, value):
        raise NotImplementedError

    @Feature.f0.setter
    def f0(self, value):
        self._set_f0(value)

    @abc.abstractmethod
    def _set_spectrum_envelope(self, value):
        raise NotImplementedError

    @Feature.spectrum_envelope.setter
    def spectrum_envelope(self, value):
        self._set_spectrum_envelope(value)

    @abc.abstractmethod
    def _set_aperiodicity(self, value):
        raise NotImplementedError

    @Feature.aperiodicity.setter
    def aperiodicity(self, value):
        self._set_aperiodicity(value)
