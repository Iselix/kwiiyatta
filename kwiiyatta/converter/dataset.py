import kwiiyatta

from . import abc


class WavFileDataset(abc.Dataset):
    def __init__(self, data_dir, Analyzer=None):
        super().__init__()
        if Analyzer is None:
            Analyzer = kwiiyatta.analyze_wav
        self.Analyzer = Analyzer
        self.data_dir = data_dir
        if not self.data_dir.exists():
            raise FileNotFoundError(f'wav files dir "{self.data_dir!s}"'
                                    f' is not found')
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f'wav files dir "{self.data_dir!s}"'
                                     f' is not directory')
        self.files = frozenset(f.relative_to(self.data_dir) for f
                               in self.data_dir.glob('*.wav'))

    def keys(self):
        return self.files

    def get_data(self, key):
        return self.Analyzer(self.data_dir/key)
