import numpy as np

import pyaudio

from scipy.io import wavfile


class Wavdata:
    def __init__(self, fs, data):
        self.fs = fs
        self.data = data

    def normalize(self, peak_lv=-1):
        dc = self.data.mean()
        self.data -= dc
        if peak_lv is not None:
            max_peak = np.power(10, peak_lv/10)
            peak = np.abs(self.data).max()
            if peak > max_peak:
                self.data *= max_peak/peak

    def save(self, wav, normalize=True, **kwargs):
        if normalize:
            self.normalize(**kwargs)
        wavfile.write(wav, self.fs, (self.data*(2**15)).astype(np.int16))

    def play(self, normalize=True, **kwargs):
        if normalize:
            self.normalize(**kwargs)
        audio = pyaudio.PyAudio()
        stream = audio.open(rate=self.fs, channels=1, format=pyaudio.paInt16,
                            output=True)
        stream.write((self.data*(2**15)).astype(np.int16),
                     num_frames=len(self.data))
        stream.close()
        audio.terminate()


def load_wav(wav):
    fs, data = wavfile.read(wav)
    if data.dtype.kind == 'f':  # float
        return Wavdata(fs, data.astype(np.float64))
    if data.dtype.kind == 'u':  # unsigned int
        assert data.dtype.itemsize == 1
        return Wavdata(fs, (data.astype(np.float64)
                            / (2**(data.dtype.itemsize*8-1)) - 1))
    assert data.dtype.kind == 'i'  # signed integer
    return Wavdata(fs, (data.astype(np.float64)
                        / (2**(data.dtype.itemsize*8-1))))
