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
