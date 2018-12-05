# Voice conversion
# by https://r9y9.github.io/nnmnkwii/v0.0.17/nnmnkwii_gallery/notebooks/vc/01-GMM%20voice%20conversion%20(en).html  # noqa

import copy
from pathlib import Path

from nnmnkwii.baseline.gmm import MLPG
from nnmnkwii.datasets import PaddedFileSourceDataset
from nnmnkwii.datasets.cmu_arctic import CMUArcticWavFileDataSource
from nnmnkwii.metrics import melcd
from nnmnkwii.preprocessing import (delta_features, remove_zeros_frames,
                                    trim_zeros_frames)
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.util import apply_each2d_trim

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

import kwiiyatta


conf = kwiiyatta.Config()
conf.add_argument('--data-root', type=str,
                  help='Root of data-set path for voice conversion')
conf.add_argument('--result-dir', type=str,
                  help='Path to write result wav files')
conf.add_argument('--gmm-seed', type=int,
                  help='Random seed for GaussianMixtureModel')
conf.parse_args()

DATA_ROOT = (Path(conf.data_root) if conf.data_root is not None
             else Path.home()/'data'/'cmu_arctic')
RESULT_ROOT = (Path(conf.result_dir) if conf.result_dir is not None
               else Path(__file__).parent/'result')

GMM_RANDOM_SEED = conf.gmm_seed

max_files = 100  # number of utterances to be used.
test_size = 0.03
use_delta = True

if use_delta:
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
else:
    windows = [
        (0, 0, np.array([1.0])),
    ]


class MyFileDataSource(CMUArcticWavFileDataSource):
    def __init__(self, *args, **kwargs):
        super(MyFileDataSource, self).__init__(*args, **kwargs)
        self.test_paths = None

    def collect_files(self):
        paths = [Path(path) for path in super(
            MyFileDataSource, self).collect_files()]
        paths_train, paths_test = train_test_split(
            paths, test_size=test_size, random_state=1234)

        # keep paths for later testing
        self.test_paths = paths_test

        return paths_train

    def collect_features(self, path):
        feature = conf.create_analyzer(path, Analyzer=kwiiyatta.analyze_wav)
        s = trim_zeros_frames(feature.spectrum_envelope)
        return feature.mel_cepstrum.data[:len(s)]  # トリムするフレームが手前にずれてるのでは？


clb_source = MyFileDataSource(data_root=DATA_ROOT,
                              speakers=["clb"], max_files=max_files)
slt_source = MyFileDataSource(data_root=DATA_ROOT,
                              speakers=["slt"], max_files=max_files)

X = PaddedFileSourceDataset(clb_source, 1200).asarray()
Y = PaddedFileSourceDataset(slt_source, 1200).asarray()
print(X.shape)
print(Y.shape)

# Alignment
X_aligned, Y_aligned = DTWAligner(verbose=0, dist=melcd).transform((X, Y))

# Drop 1st (power) dimension
X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]

static_dim = X_aligned.shape[-1]
if use_delta:
    X_aligned = apply_each2d_trim(delta_features, X_aligned, windows)
    Y_aligned = apply_each2d_trim(delta_features, Y_aligned, windows)

XY = (np.concatenate((X_aligned, Y_aligned), axis=-1)
        .reshape(-1, X_aligned.shape[-1]*2))
print(XY.shape)

XY = remove_zeros_frames(XY)
print(XY.shape)

gmm = GaussianMixture(
    n_components=64, covariance_type="full", max_iter=100, verbose=1,
    random_state=GMM_RANDOM_SEED
)

gmm.fit(XY)


def test_one_utt(src_path, tgt_path, disable_mlpg=False, diffvc=True):
    # GMM-based parameter generation is provided by the library
    # in `baseline` module
    if disable_mlpg:
        # Force disable MLPG
        paramgen = MLPG(gmm, windows=[(0, 0, np.array([1.0]))], diff=diffvc)
    else:
        paramgen = MLPG(gmm, windows=windows, diff=diffvc)

    src = conf.create_analyzer(src_path, Analyzer=kwiiyatta.analyze_wav)

    mcep = copy.copy(src.mel_cepstrum)
    c0, mc = mcep.data[:, 0], mcep.data[:, 1:]
    if use_delta:
        mc = delta_features(mc, windows)
    mc = paramgen.transform(mc)
    if disable_mlpg and mc.shape[-1] != static_dim:
        mc = mc[:, :static_dim]
    assert mc.shape[-1] == static_dim
    mcep.data = np.hstack((c0[:, None], mc))
    if diffvc:
        wav = kwiiyatta.apply_mlsa_filter(src, mcep)
    else:
        feature = kwiiyatta.feature(src)
        feature.mel_cepstrum = mcep
        wav = feature.synthesize()

    return wav


for i, (src_path, tgt_path) in enumerate(zip(clb_source.test_paths,
                                             slt_source.test_paths)):
    print("{}-th sample".format(i+1))
    diff_MLPG = test_one_utt(src_path, tgt_path)
    synth_MLPG = test_one_utt(src_path, tgt_path, diffvc=False)

    result_path = RESULT_ROOT/src_path.name
    result_path.parent.mkdir(parents=True, exist_ok=True)

    print("diff MLPG")
    diff_MLPG.save(result_path.with_suffix('.diff.wav'))
    print("synth MLPG")
    synth_MLPG.save(result_path.with_suffix('.synth.wav'))
