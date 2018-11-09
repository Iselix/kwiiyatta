import numpy as np

from scipy.io import wavfile


class Wavdata:
    def __init__(self, fs, data):
        self.fs = fs
        self.data = data

    def save(self, wav):
        wavfile.write(wav, self.fs, self.data.astype(np.int16))


def load_wav(wav):
    fs, data = wavfile.read(wav)
    return Wavdata(fs, data.astype(np.float64))
