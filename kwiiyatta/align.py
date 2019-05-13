import kwiiyatta

from . import converter
from . import vocoder


def align(a, b, **kwargs):
    if isinstance(a, vocoder.abc.Feature):
        if not isinstance(b, vocoder.abc.Feature):
            raise TypeError(f'argument type mismatch: {type(a)!r}'
                            f' and {type(b)!r}')
        return vocoder.align(a, b, **kwargs)
    elif isinstance(a, converter.abc.Dataset):
        if not isinstance(b, converter.abc.Dataset):
            raise TypeError(f'argument type mismatch: {type(a)!r}'
                            f' and {type(b)!r}')
        return kwiiyatta.align_dataset(kwiiyatta.ParallelDataset(a, b))
    else:
        raise TypeError('argument should be Feature or Dataset')
