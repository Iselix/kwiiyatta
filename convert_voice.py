# Voice conversion
# by https://r9y9.github.io/nnmnkwii/v0.0.17/nnmnkwii_gallery/notebooks/vc/01-GMM%20voice%20conversion%20(en).html  # noqa

from pathlib import Path

import kwiiyatta


conf = kwiiyatta.Config()
conf.add_argument('--result-dir', type=str,
                  help='Path to write result wav files')
conf.add_argument('files', type=str, nargs='+',
                  help='Wav files to convert voice')
conf.add_converter_arguments()
conf.parse_args()

RESULT_ROOT = (Path(conf.result_dir) if conf.result_dir is not None
               else Path(__file__).parent/'result')

max_files = 100  # number of utterances to be used.
test_size = 0.03
use_delta = True

converter = conf.train_converter(use_delta=use_delta)


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


for i, conv_file in enumerate(conf.files):
    conv_path = Path(conv_file)
    print("{}-th sample".format(i+1))
    diff_MLPG = test_one_utt(conv_path)
    synth_MLPG = test_one_utt(conv_path, diffvc=False)

    result_path = conv_path
    if conf.result_dir is not None:
        result_path = Path(conf.result_dir)/conv_path.name
        result_path.parent.mkdir(parents=True, exist_ok=True)

    print("diff MLPG")
    diff_MLPG.save(result_path.with_suffix('.diff.wav'))
    print("synth MLPG")
    synth_MLPG.save(result_path.with_suffix('.synth.wav'))
