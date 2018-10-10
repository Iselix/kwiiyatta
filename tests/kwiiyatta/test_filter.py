import copy

import numpy as np

import pytest

import kwiiyatta

from tests import dataset, feature
from tests.plugin import assert_any


@pytest.mark.assert_any
@pytest.mark.parametrize('fs2', [16000, 44100])
@pytest.mark.parametrize('fs1', dataset.FS)
def test_mlsa_filter(fs1, fs2):
    np.random.seed(0)
    clb = feature.get_analyzer(dataset.get_wav_path(dataset.CLB_WAV, fs=fs1))
    slt = feature.get_analyzer(dataset.get_wav_path(dataset.SLT_WAV, fs=fs2))
    slt_aligned = kwiiyatta.align(slt, clb)

    mcep_diff = copy.copy(slt_aligned.mel_cepstrum)
    mcep_diff.data = mcep_diff.data - clb.mel_cepstrum.resample_data(slt.fs)
    result = kwiiyatta.apply_mlsa_filter(clb.wavdata, mcep_diff)

    expected = kwiiyatta.feature(clb)
    if clb.fs > slt.fs:
        slt_shape = clb.spectrum_len * slt.fs // clb.fs
        expected.spectrum_envelope = \
            np.hstack((
                feature.override_power(
                    slt_aligned.reshaped_spectrum_envelope(slt_shape),
                    clb.spectrum_envelope[:, :slt_shape]
                ),
                clb.spectrum_envelope[:, slt_shape:]
            ))
    else:
        expected.spectrum_envelope = \
            slt_aligned.resample_spectrum_envelope(clb.fs)
        feature.override_spectrum_power(expected, clb)

    actual = kwiiyatta.Analyzer(result)
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, actual)
    assert_any.between(0.064, f0_diff, 0.097)
    assert_any.between(0.31, spec_diff, 0.54)
    assert_any.between(0.035, ape_diff, 0.081)
    assert_any.between(0.036, mcep_diff, 0.086)
