from .align import align, align_even
from .feature import Feature, feature
from .mcep import MelCepstrum
from .world import WorldAnalyzer, WorldSynthesizer


Analyzer = WorldAnalyzer
Synthesizer = WorldSynthesizer


def analyze_wav(wavfile, Analyzer=None, **kwargs):
    def default_analyzer(DefaultAnalyzer):
        if DefaultAnalyzer is None:
            global Analyzer
            return Analyzer
        return DefaultAnalyzer

    return default_analyzer(Analyzer).load_wav(wavfile, **kwargs)


def reshape(feature, reshape_spectrum_len):
    f = Feature.init(feature)
    if feature.spectrum_len != reshape_spectrum_len:
        f.reshape(reshape_spectrum_len)
    return f


__all__ = ['analyze_wav', 'reshape']
__all__ += ['align', 'align_even']
__all__ += ['Feature', 'feature']
__all__ += ['MelCepstrum']
__all__ += ['Analyzer', 'Synthesizer']
