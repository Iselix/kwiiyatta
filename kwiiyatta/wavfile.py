import numpy as np

import pyaudio

from scipy.io import wavfile


class Wavdata:
    def __init__(self, fs, data):
        self.fs = fs
        self.data = data

    def save(self, wav):
        wavfile.write(wav, self.fs, self.data.astype(np.int16))

    def play(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(rate=self.fs, channels=1, format=pyaudio.paInt16,
                            output=True)
        stream.write(self.data.astype(np.int16), num_frames=len(self.data))
        stream.close()
        audio.terminate()


def load_wav(wav):
    fs, data = wavfile.read(wav)
    return Wavdata(fs, data.astype(np.float64))
