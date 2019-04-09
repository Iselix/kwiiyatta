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


def get_dataset_path(dataset_path, fullset=False):
    if fullset:
        import tests
        dataset_path = FULLSET_ROOT/(dataset_path.relative_to(DATASET_ROOT))
        if not dataset_path.exists():
            pytest.skip(f'fullset of {str(dataset_path.name)!r} is not exists')
        if tests._skip_fullset:
            pytest.skip(f'need --run-fullset option to run fullset tests')

    return dataset_path


def get_wav_path(wav_path, fullset=False):
    dataset_path = wav_path.parent
    wav_path = wav_path.relative_to(dataset_path)
    dataset_path = get_dataset_path(dataset_path, fullset)

    result = dataset_path/wav_path
    assert result.is_file, 'wav file not found: {result!s}'
    return result
