import copy

import kwiiyatta

from tests import dataset, feature


def test_mlsa_filter(check):
    clb = feature.get_analyzer(dataset.CLB_WAV)
    slt = feature.get_analyzer(dataset.SLT_WAV)
    slt_aligned = kwiiyatta.align(slt, clb)

    mcep_diff = copy.copy(slt_aligned.mel_cepstrum)
    mcep_diff.data -= clb.mel_cepstrum.data
    result = kwiiyatta.apply_mlsa_filter(clb.wavdata, mcep_diff)

    expected = kwiiyatta.feature(clb)
    expected.spectrum_envelope = slt_aligned.spectrum_envelope
    feature.override_spectrum_power(expected, clb)

    actual = kwiiyatta.Analyzer(result)
    f0_diff, spec_diff, ape_diff, mcep_diff = \
        feature.calc_feature_diffs(expected, actual)
    check.round_equal(0.070, f0_diff)
    check.round_equal(0.32, spec_diff)
    check.round_equal(0.080, ape_diff)
    check.round_equal(0.12, mcep_diff)
