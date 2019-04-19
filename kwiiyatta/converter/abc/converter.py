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
