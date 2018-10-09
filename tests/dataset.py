import itertools
import pathlib

import pytest


TEST_ROOT = pathlib.Path(__file__).parent
DATASET_ROOT = TEST_ROOT/'data'
CLB_ROOT = DATASET_ROOT/'cmu_us_clb_arctic'
SLT_ROOT = DATASET_ROOT/'cmu_us_slt_arctic'

CLB_DIR = CLB_ROOT/'wav'
CLB_WAV = CLB_DIR/'arctic_a0001.wav'
CLB_WAV2 = CLB_DIR/'arctic_a0002.wav'

SLT_DIR = SLT_ROOT/'wav'
SLT_WAV = SLT_DIR/'arctic_a0001.wav'
SLT_WAV2 = SLT_DIR/'arctic_a0002.wav'

FULLSET_ROOT = DATASET_ROOT/'fullset'

DTYPES = ['u8', 'i16', 'i32', 'f32', 'f64']
FS = [16000, 22050, 44100, 48000, 96000]
FS_COMB = [(fs1, fs2) for fs1, fs2 in itertools.combinations(FS, 2)]


def get_dataset_path(wavdir_path, fullset=False, dtype='i16', fs=16000):
    dataset_path = wavdir_path.parent
    suffix = ''
    if dtype == 'i16':
        pass
    else:
        suffix = f'.{dtype!s}'

    if fs == 16000:
        pass
    elif fs == 22050:
        suffix += '.22'
    elif fs == 44100:
        suffix += '.44'
    elif fs == 48000:
        suffix += '.48'
    elif fs == 96000:
        suffix += '.96'
    else:
        assert False, f'fs {fs!s} hz is not supported'

    dataset_path = dataset_path.with_suffix(suffix)

    if fullset:
        import tests
        dataset_path = FULLSET_ROOT/(dataset_path.relative_to(DATASET_ROOT))
        if not dataset_path.exists():
            pytest.skip(f'fullset of {str(dataset_path.name)!r} is not exists')
        if tests._skip_fullset:
            pytest.skip(f'need --run-fullset option to run fullset tests')

    dataset_path = dataset_path/wavdir_path.name

    assert dataset_path.is_dir(), f'dataset {dataset_path!s} is not found'
    return dataset_path


def get_wav_path(wav_path, fullset=False, dtype='i16', fs=16000):
    dataset_path = wav_path.parent
    wav_path = wav_path.relative_to(dataset_path)
    dataset_path = get_dataset_path(dataset_path, fullset, dtype, fs)

    result = dataset_path/wav_path
    assert result.is_file, 'wav file not found: {result!s}'
    return result
