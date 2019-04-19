import copy

import pytest

import kwiiyatta
import kwiiyatta.converter.abc as converter_abc
from kwiiyatta.converter import (DeltaFeatureConverter, MelCepstrumDataset,
                                 MelCepstrumFeatureConverter)

from tests import dataset, feature


class NopConverter(converter_abc.FeatureConverter):
    def _train(self, dataarray):
        pass

    def convert(self, feature):
        return feature


def test_delta_converter():
    a = feature.get_analyzer(dataset.CLB_WAV)
    base = {'key': a}
    delta_converter = DeltaFeatureConverter(NopConverter())
    delta_converter.train(MelCepstrumDataset(base), ['key'])

    with pytest.raises(ValueError) as e:
        a_fp3 = feature.get_analyzer(dataset.CLB_WAV, frame_period=3)
        delta_converter.convert(a_fp3.mel_cepstrum.data[:, 1:], a_fp3)

    assert 'frame_period is expected to 5 but 3' == str(e.value)

    mcep = a.mel_cepstrum.data[:, 1:]
    assert (mcep == delta_converter.convert(mcep, a)).all()


def test_mcep_converter():
    a = kwiiyatta.feature(feature.get_analyzer(dataset.CLB_WAV))
    base = {'key': copy.copy(a)}
    mcep_converter = MelCepstrumFeatureConverter(NopConverter())
    mcep_converter.train(base, ['key'])
    mcep = feature.get_analyzer(dataset.CLB_WAV).mel_cepstrum

    with pytest.raises(ValueError) as e:
        a.mel_cepstrum_order = 32
        mcep_converter.convert(a.mel_cepstrum)

    assert 'order is expected to 24 but 32' == str(e.value)

    with pytest.raises(ValueError) as e:
        a.mel_cepstrum_order = 24
        mcep = a.mel_cepstrum
        mcep._fs = 44100
        mcep_converter.convert(mcep)

    assert 'fs is expected to 16000 but 44100' == str(e.value)
