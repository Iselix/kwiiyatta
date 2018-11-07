import abc


class Synthesizer(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def synthesize(feature):
        raise NotImplementedError
