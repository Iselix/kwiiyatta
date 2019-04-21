import numpy as np

import pysptk
from pysptk.synthesis import MLSADF, Synthesizer

import kwiiyatta


def apply_mlsa_filter(wav, mcep):
    if wav.fs != mcep.fs:
        raise ValueError(f'wav fs ({wav.fs!r}) and mcep fs ({mcep.fs!r} is '
                         f'mismatch)')
    # remove power coefficients
    mc = np.hstack((np.zeros((mcep.data.shape[0], 1)), mcep.data[:, 1:]))
    alpha = mcep.alpha()
    engine = Synthesizer(MLSADF(order=mcep.order, alpha=alpha),
                         hopsize=int(mcep.fs * (mcep.frame_period * 0.001)))
    b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
    waveform = engine.synthesis(wav.data, b)
    return kwiiyatta.Wavdata(wav.fs, waveform)
