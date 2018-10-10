import numpy as np

import pysptk
from pysptk.synthesis import MLSADF, Synthesizer

import kwiiyatta


def apply_mlsa_filter(wav, mcep):
    if mcep.fs > wav.fs:
        mcep = kwiiyatta.resample(mcep, wav.fs)
    elif mcep.fs < wav.fs:
        spec = kwiiyatta.Synthesizer.resample_spectrum_envelope(
            mcep.extract_spectrum(),
            mcep.fs,
            wav.fs
        )
        cutoff = mcep.fs*spec.shape[1]//wav.fs
        spec[:, cutoff:] = np.tile(np.atleast_2d(spec[:, cutoff-1]).T,
                                   spec.shape[-1]-cutoff)
        mcep = kwiiyatta.MelCepstrum(wav.fs, mcep.frame_period)
        mcep.extract(spec)
    # remove power coefficients
    mc = np.hstack((np.zeros((mcep.data.shape[0], 1)), mcep.data[:, 1:]))
    alpha = mcep.alpha()
    engine = Synthesizer(MLSADF(order=mcep.order, alpha=alpha),
                         hopsize=int(mcep.fs * (mcep.frame_period * 0.001)))
    b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
    waveform = engine.synthesis(wav.data, b)
    return kwiiyatta.Wavdata(wav.fs, waveform)
