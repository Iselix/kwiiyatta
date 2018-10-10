import pysptk

import kwiiyatta


class MelCepstrum:
    def __init__(self, fs, frame_period, data=None):
        self._fs = fs
        self._frame_period = frame_period
        self.data = data

    @property
    def fs(self):
        return self._fs

    @property
    def frame_period(self):
        return self._frame_period

    @property
    def order(self):
        return self.data.shape[-1] - 1

    @staticmethod
    def fs_alpha(fs):
        return pysptk.util.mcepalpha(fs)

    def alpha(self):
        return self.fs_alpha(self.fs)

    def resample_data(self, new_fs, spectrum_len=None, Synthesizer=None,
                      order=None):
        if Synthesizer is None:
            Synthesizer = kwiiyatta.Synthesizer
        if spectrum_len is None:
            spectrum_len = Synthesizer.fs_spectrum_len(self.fs)
        if order is None:
            order = self.order

        spec = Synthesizer.resample_spectrum_envelope(
            self.extract_spectrum(spectrum_len),
            self.fs,
            new_fs
        )
        return self.extract_data(spec, order, fs=new_fs)

    def resample(self, new_fs, spectrum=None, order=None, **kwargs):
        if order is None:
            order = self.order
        if spectrum is not None:
            self._fs = new_fs
            if order is not None:
                kwargs['order'] = order
            self.extract(spectrum, **kwargs)
        elif new_fs != self.fs:
            self.data = self.resample_data(new_fs, order=order,
                                           **kwargs)
            self._fs = new_fs

    def extract_spectrum(self, spectrum_len=None, Synthesizer=None):
        if spectrum_len is None:
            if Synthesizer is None:
                Synthesizer = kwiiyatta.Synthesizer
            spectrum_len = Synthesizer.fs_spectrum_len(self.fs)
        return pysptk.mc2sp(self.data, fftlen=(spectrum_len - 1) * 2,
                            alpha=self.alpha())

    def extract_data(self, spectrum, order=24, fs=None):
        if fs is None:
            fs = self.fs
        return pysptk.sp2mc(spectrum, order=order, alpha=self.fs_alpha(fs))

    def extract(self, spectrum, order=24):
        self.data = self.extract_data(spectrum, order)
        return self.data
