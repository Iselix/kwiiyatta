from .feature import Feature, feature
from .world import WorldAnalyzer


Analyzer = WorldAnalyzer


def analyze_wav(wavfile, Analyzer=None, **kwargs):
    def default_analyzer(DefaultAnalyzer):
        if DefaultAnalyzer is None:
            global Analyzer
            return Analyzer
        return DefaultAnalyzer

    return default_analyzer(Analyzer).load_wav(wavfile, **kwargs)


__all__ = ['analyze_wav']
__all__ += ['Feature', 'feature']
__all__ += ['Analyzer']
