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


def get_dataset_path(wavdir_path, fullset=False, dtype='i16'):
    dataset_path = wavdir_path.parent
    suffix = ''
    if dtype == 'i16':
        pass
    else:
        suffix = f'.{dtype!s}'
    dataset_path = dataset_path.with_suffix(suffix)

    if fullset:
        import tests
        dataset_path = FULLSET_ROOT/(dataset_path.relative_to(DATASET_ROOT))
        if not dataset_path.exists():
            pytest.skip(f'fullset of {str(dataset_path.name)!r} is not exists')
        if tests._skip_fullset:
            pytest.skip(f'need --run-fullset option to run fullset tests')

    dataset_path = dataset_path/wavdir_path.name

    assert dataset_path.is_dir(), f'dtype {dtype!s} is not supported'
    return dataset_path


def get_wav_path(wav_path, fullset=False, dtype='i16'):
    dataset_path = wav_path.parent
    wav_path = wav_path.relative_to(dataset_path)
    dataset_path = get_dataset_path(dataset_path, fullset, dtype)

    result = dataset_path/wav_path
    assert result.is_file, 'wav file not found: {result!s}'
    return result
