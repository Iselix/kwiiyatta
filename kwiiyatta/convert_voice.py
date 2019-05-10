import pathlib

import kwiiyatta


def main():
    conf = kwiiyatta.Config()
    conf.add_argument('--result-dir', type=str,
                      help='Path to write result wav files')
    conf.add_argument('files', type=str, nargs='+',
                      help='Wav files to convert voice')
    conf.add_converter_arguments()
    conf.parse_args()

    converter = conf.train_converter(use_delta=True)

    for conv_file in conf.files:
        conv_path = pathlib.Path(conv_file)
        diff_MLPG = convert(conf, converter, conv_path)
        synth_MLPG = convert(conf, converter, conv_path, diffvc=False)

        result_path = conv_path
        if conf.result_dir is not None:
            result_path = pathlib.Path(conf.result_dir)/conv_path.name
            result_path.parent.mkdir(parents=True, exist_ok=True)

        result = result_path.with_suffix('.diff.wav')
        print(f'diff MLPG: {result!s}')
        diff_MLPG.save(result)
        result = result_path.with_suffix('.synth.wav')
        print(f'synth MLPG: {result!s}')
        synth_MLPG.save(result)


def convert(conf, converter, src_path, diffvc=True):
    src = conf.create_analyzer(src_path, Analyzer=kwiiyatta.analyze_wav)

    mcep = converter.convert(src.mel_cepstrum, diff=diffvc)
    if diffvc:
        wav = kwiiyatta.apply_mlsa_filter(src, mcep)
    else:
        feature = kwiiyatta.feature(src)
        feature.mel_cepstrum = mcep
        wav = feature.synthesize()

    return wav


if __name__ == '__main__':
    main()
