import abc


class Synthesizer(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def synthesize(feature):
        raise NotImplementedError

    @staticmethod
    def fs_spectrum_len(fs):
        raise NotImplementedError
