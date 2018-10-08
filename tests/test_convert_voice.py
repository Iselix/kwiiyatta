import pathlib
import subprocess

import pytest


pytestmark = pytest.mark.slow


TEST_ROOT = pathlib.Path(__file__).parent
DATASET_ROOT = TEST_ROOT/'data'


def test_voice_conversion(tmpdir):
    result_root = pathlib.Path(tmpdir)
    subprocess.run(
        [
            'python', 'convert_voice.py',
            '--data-root', str(DATASET_ROOT),
            '--result-dir', str(result_root),
        ], check=True)

    assert (result_root/'arctic_a0009.diff.wav').is_file()
    assert (result_root/'arctic_a0009.synth.wav').is_file()
