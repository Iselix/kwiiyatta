import abc
import copy

import numpy as np

import kwiiyatta

from ..mcep import MelCepstrum


class Feature(abc.ABC):
    def __init__(self, fs, frame_period=5, mcep_order=24, Synthesizer=None):
        self.mel_cepstrum_order = mcep_order
        if Synthesizer is None:
            self.Synthesizer = kwiiyatta.Synthesizer
        else:
            self.Synthesizer = Synthesizer
        self._mel_cepstrum = MelCepstrum(fs, frame_period)
        self._is_voiced = None

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._mel_cepstrum = copy.copy(self._mel_cepstrum)
        return result

    @property
    def fs(self):
        return self._mel_cepstrum.fs

    @property
    def frame_period(self):
        return self._mel_cepstrum.frame_period

    @property
    def frame_len(self):
        return self.f0.shape[0]

    @property
    @abc.abstractmethod
    def spectrum_len(self):
        return self.Synthesizer.fs_spectrum_len(self.fs)

    @abc.abstractmethod
    def _get_f0(self):
        raise NotImplementedError

    def reshaped_spectrum_envelope(self, new_spectrum_len):
        spec = self._get_spectrum_envelope()
        if spec is None:
            return self._mel_cepstrum.extract_spectrum(new_spectrum_len)
        return self.Synthesizer.reshape_spectrum_envelope(
            spec, self.fs, new_spectrum_len)

    def reshaped_aperiodicity(self, new_spectrum_len):
        return self.Synthesizer.reshape_aperiodicity(
            self.aperiodicity, self.fs, new_spectrum_len)

    def resample_spectrum_envelope(self, new_fs):
        return self.Synthesizer.resample_spectrum_envelope(
            self.spectrum_envelope,
            self.fs,
            new_fs,
        )

    def resample_aperiodicity(self, new_fs):
        return self.Synthesizer.resample_aperiodicity(
            self.aperiodicity,
            self.fs,
            new_fs
        )

    def resample_mel_cepstrum(self, new_fs):
        if self._get_spectrum_envelope() is not None:
            return kwiiyatta.resample(
                self.mel_cepstrum, new_fs,
                order=self.mel_cepstrum_order,
                spectrum=self.resample_spectrum_envelope(new_fs)
            )
        return kwiiyatta.resample(
            self.mel_cepstrum, new_fs,
            order=self.mel_cepstrum_order,
            Synthesizer=self.Synthesizer
        )

    @property
    def f0(self):
        return self._get_f0()

    def extract_spectrum_envelope(self, spectrum_len=None):
        if self._mel_cepstrum.data is not None:
            if spectrum_len is None:
                spectrum_len = self.spectrum_len
            return self._mel_cepstrum.extract_spectrum(spectrum_len)
        return None

    @abc.abstractmethod
    def _get_spectrum_envelope(self):
        raise NotImplementedError

    @property
    def spectrum_envelope(self):
        spec = self._get_spectrum_envelope()
        if spec is not None:
            return spec
        return self.extract_spectrum_envelope()

    @abc.abstractmethod
    def _get_aperiodicity(self):
        raise NotImplementedError

    @property
    def aperiodicity(self):
        return self._get_aperiodicity()

    def extract_mel_cepstrum(self, spectrum=None):
        if spectrum is not None:
            self._set_spectrum_envelope(None)
            spec = spectrum
        else:
            if self._mel_cepstrum.data is not None:
                if self.mel_cepstrum_order == self._mel_cepstrum.order:
                    return self._mel_cepstrum
                elif self.mel_cepstrum_order < self._mel_cepstrum.order:
                    mcep = copy.copy(self._mel_cepstrum)
                    mcep.data = mcep.data[:, :self.mel_cepstrum_order+1]
                    return mcep
            spec = self.spectrum_envelope

        if spec is not None:
            self._mel_cepstrum.extract(spec, self.mel_cepstrum_order)
            return self._mel_cepstrum
        return None

    def clear_mel_cepstrum(self):
        self._mel_cepstrum.data = None

    @property
    def mel_cepstrum(self):
        return self.extract_mel_cepstrum()

    def extract_is_voiced(self):
        if self._is_voiced is None:
            self._is_voiced = self.Synthesizer.extract_is_voiced(self)
        return self._is_voiced

    @property
    def is_voiced(self):
        return self.extract_is_voiced()

    @abc.abstractmethod
    def ascontiguousarray(self):
        raise NotImplementedError

    def synthesize(self, **kwargs):
        return self.Synthesizer.synthesize(self, **kwargs)

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

    def __getitem__(self, key):
        f0 = self.f0[key]
        spec = self.spectrum_envelope[key]
        ape = self.aperiodicity[key]
        is_voiced = self._is_voiced
        if is_voiced is not None:
            is_voiced = is_voiced[key]
        mcep = None
        if self._mel_cepstrum.data is not None:
            mcep = self.mel_cepstrum.data[key]

        if isinstance(key, int):
            if mcep is None:
                mcep = self.mel_cepstrum.data[key]
            return f0, spec, ape, mcep

        result = kwiiyatta.feature(self)
        result._f0 = f0
        result._spectrum_envelope = spec
        result._aperiodicity = ape
        if is_voiced is not None:
            result._is_voiced = is_voiced
        if mcep is not None:
            result._mel_cepstrum.data = mcep
        return result


class MutableFeature(Feature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reshape(self, new_spectrum_len):
        if self._get_spectrum_envelope().shape[1] != new_spectrum_len:
            self._set_spectrum_envelope(
                self.reshaped_spectrum_envelope(new_spectrum_len))
        if self._get_aperiodicity().shape[1] != new_spectrum_len:
            self._set_aperiodicity(
                self.reshaped_aperiodicity(new_spectrum_len))

    def resample(self, new_fs):
        if new_fs != self.fs:
            # f0 に関しては何もしない
            if self._get_aperiodicity() is not None:
                self._set_aperiodicity(
                    self.resample_aperiodicity(new_fs))
            if self._get_spectrum_envelope() is not None:
                self._set_spectrum_envelope(
                    self.resample_spectrum_envelope(new_fs))
                self._mel_cepstrum.data = None
            elif self._mel_cepstrum.data is not None:
                self._mel_cepstrum.resample(new_fs,
                                            Synthesizer=self.Synthesizer)
            self._mel_cepstrum._fs = new_fs

    @abc.abstractmethod
    def _set_f0(self, value):
        raise NotImplementedError

    @Feature.f0.setter
    def f0(self, value):
        if value is not None:
            self._is_voiced = None
        self._set_f0(value)

    @abc.abstractmethod
    def _set_spectrum_envelope(self, value):
        raise NotImplementedError

    @Feature.spectrum_envelope.setter
    def spectrum_envelope(self, value):
        if value is None:
            if self._mel_cepstrum.data is None:
                self.extract_mel_cepstrum()
        else:
            self._mel_cepstrum.data = None
            self._is_voiced = None
        self._set_spectrum_envelope(value)

    @abc.abstractmethod
    def _set_aperiodicity(self, value):
        raise NotImplementedError

    @Feature.aperiodicity.setter
    def aperiodicity(self, value):
        if value is not None:
            self._is_voiced = None
        self._set_aperiodicity(value)

    @Feature.mel_cepstrum.setter
    def mel_cepstrum(self, data):
        if data is not None:
            if isinstance(data, MelCepstrum):
                if data.fs != self.fs:
                    data = data.resample_data(self.fs,
                                              Synthesizer=self.Synthesizer)
                else:
                    data = data.data
            elif not isinstance(data, np.ndarray):
                raise TypeError('Feature.mel_cepstrum should be a MelCepstrum'
                                ' or ndarray')
            if data is not None:
                self._set_spectrum_envelope(None)
        self._mel_cepstrum.data = data

    def ascontiguousarray(self):
        self._set_f0(
            np.ascontiguousarray(self.f0))
        self._set_spectrum_envelope(
            np.ascontiguousarray(self.spectrum_envelope))
        self._set_aperiodicity(
            np.ascontiguousarray(self.aperiodicity))
