# Voice conversion
# by https://r9y9.github.io/nnmnkwii/v0.0.17/nnmnkwii_gallery/notebooks/vc/01-GMM%20voice%20conversion%20(en).html  # noqa

from pathlib import Path

from sklearn.model_selection import train_test_split

import kwiiyatta


conf = kwiiyatta.Config()
conf.add_argument('--data-root', type=str,
                  help='Root of data-set path for voice conversion')
conf.add_argument('--result-dir', type=str,
                  help='Path to write result wav files')
conf.add_converter_arguments()
conf.parse_args()

DATA_ROOT = (Path(conf.data_root) if conf.data_root is not None
             else Path.home()/'data'/'cmu_arctic')
RESULT_ROOT = (Path(conf.result_dir) if conf.result_dir is not None
               else Path(__file__).parent/'result')

max_files = 100  # number of utterances to be used.
test_size = 0.03
use_delta = True


def train_and_test_paths(keys):
    paths = sorted(keys)[:max_files]
    return train_test_split(paths, test_size=test_size, random_state=1234)


src_dataset = kwiiyatta.WavFileDataset(DATA_ROOT/'cmu_us_clb_arctic'/'wav')
tgt_dataset = kwiiyatta.WavFileDataset(DATA_ROOT/'cmu_us_slt_arctic'/'wav')

dataset = kwiiyatta.align(src_dataset, tgt_dataset)

train_paths, test_paths = train_and_test_paths(dataset.keys())

converter = conf.create_converter(use_delta=use_delta)

converter.train(dataset, train_paths)


def test_one_utt(src_path, disable_mlpg=False, diffvc=True):
    src = conf.create_analyzer(src_path, Analyzer=kwiiyatta.analyze_wav)

    mcep = converter.convert(src.mel_cepstrum, mlpg=not disable_mlpg,
                             diff=diffvc)
    if diffvc:
        wav = kwiiyatta.apply_mlsa_filter(src, mcep)
    else:
        feature = kwiiyatta.feature(src)
        feature.mel_cepstrum = mcep
        wav = feature.synthesize()

    return wav


for i, src_path in enumerate(test_paths):
    src_path = DATA_ROOT/'cmu_us_clb_arctic'/'wav'/src_path
    print("{}-th sample".format(i+1))
    diff_MLPG = test_one_utt(src_path)
    synth_MLPG = test_one_utt(src_path, diffvc=False)

    result_path = RESULT_ROOT/src_path.name
    result_path.parent.mkdir(parents=True, exist_ok=True)

    print("diff MLPG")
    diff_MLPG.save(result_path.with_suffix('.diff.wav'))
    print("synth MLPG")
    synth_MLPG.save(result_path.with_suffix('.synth.wav'))
