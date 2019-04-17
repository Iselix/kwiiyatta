import abc
import collections.abc


class Dataset(collections.abc.Mapping):
    @abc.abstractmethod
    def keys(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(self, key):
        raise NotImplementedError

    def __getitem__(self, key):
        return self.get_data(key)

    def __iter__(self):
        return ((key, self[key]) for key in self.keys())

    def __len__(self):
        return len(self.keys())
