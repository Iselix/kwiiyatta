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


class MapDataset(Dataset):
    expand_tuple = True
    with_key = False
    with_raw = False

    def __init__(self, base_dataset, **kwargs):
        super().__init__()
        self.base = base_dataset
        self.kwargs = kwargs

    def keys(self):
        return self.base.keys()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_data(self, key, with_raw=False):
        if isinstance(self.base, MapDataset):
            data, raw = self.base.get_data(key, with_raw=True)
        else:
            data = self.base[key]
            raw = data

        args = {**self.kwargs}
        if self.with_key:
            args['key'] = key

        if self.expand_tuple and isinstance(data, tuple):
            if self.with_raw:
                result = tuple(self.function(d, raw=r, **args) for d, r
                               in zip(data, raw))
            else:
                result = tuple(self.function(d, **args) for d in data)
        else:
            if self.with_raw:
                args['raw'] = raw
            result = self.function(data, **args)

        if with_raw:
            return result, raw
        return result

    @staticmethod
    @abc.abstractmethod
    def function(data):
        raise NotImplementedError


def map_dataset(expand_tuple=True, with_key=False, with_raw=False):
    return lambda func: (
        type(func.__name__, (MapDataset,),
             {
                 '__module__': func.__module__, '__doc__': func.__doc__,
                 'function': staticmethod(func),
                 'expand_tuple': expand_tuple,
                 'with_key': with_key,
                 'with_raw': with_raw,
             }))
