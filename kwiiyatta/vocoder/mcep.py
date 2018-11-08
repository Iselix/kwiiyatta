import pysptk


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

    def extract_spectrum(self, spectrum_len):
        return pysptk.mc2sp(self.data, fftlen=(spectrum_len - 1) * 2,
                            alpha=self.alpha())

    def extract(self, spectrum, order=24):
        self.data = pysptk.sp2mc(spectrum, order=order, alpha=self.alpha())
        return self.data
