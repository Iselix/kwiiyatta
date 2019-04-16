import pathlib
import subprocess

import pytest

from tests import dataset


pytestmark = pytest.mark.slow


def test_voice_conversion(tmpdir):
    result_root = pathlib.Path(tmpdir)
    subprocess.run(
        [
            'python', 'convert_voice.py',
            '--data-root', str(dataset.DATASET_ROOT),
            '--result-dir', str(result_root),
            '--gmm-seed', '0',
        ], check=True)

    assert (result_root/'arctic_a0009.diff.wav').is_file()
    assert (result_root/'arctic_a0009.synth.wav').is_file()


def test_voice_conversion_fullset(tmpdir):
    result_root = pathlib.Path(tmpdir)
    subprocess.run(
        [
            'python', 'convert_voice.py',
            '--data-root', str(dataset.get_dataset_path(dataset.DATASET_ROOT,
                                                        fullset=True)),
            '--result-dir', str(result_root),
            '--gmm-seed', '0',
        ], check=True)

    results = ['arctic_a0036', 'arctic_a0041', 'arctic_a0082']

    for result in results:
        result_path = (result_root/result).with_suffix('.diff.wav')
        assert result_path.is_file()

        result_path = (result_root/result).with_suffix('.synth.wav')
        assert result_path.is_file()
