import pytest

import kwiiyatta.converter.abc as converter_abc
from kwiiyatta.converter import DeltaFeatureConverter, MelCepstrumDataset

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
