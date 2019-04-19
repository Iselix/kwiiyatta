import abc


class FeatureConverter(abc.ABC):
    @abc.abstractmethod
    def _train(self, dataarray, **kwargs):
        raise NotImplementedError

    def train(self, dataset, keys, **kwargs):
        from .. import dataset as ds
        self._train(ds.make_dataset_to_array(dataset, keys), **kwargs)

    @abc.abstractmethod
    def convert(self, feature, **kwargs):
        raise NotImplementedError


class MapFeatureConverter(FeatureConverter):
    def __init__(self, base_converter):
        self.base = base_converter

    def _train(self, dataarray, **kwargs):
        return self.base._train(dataarray, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)

    @abc.abstractmethod
    def convert(self, feature, raw=None, **kwargs):
        if isinstance(self.base, MapFeatureConverter):
            return self.base.convert(feature, raw, **kwargs)
        return self.base.convert(feature, **kwargs)
