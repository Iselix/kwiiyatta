import pathlib

import kwiiyatta


def main():
    conf = kwiiyatta.Config()
    conf.add_argument('source', type=str,
                      help='Source wav file of voice resynthesis')
    conf.add_argument('--result-dir', type=str,
                      help='Path to write result wav files')
    conf.parse_args()

    source_path = pathlib.Path(conf.source).resolve()
    source = conf.create_analyzer(source_path, Analyzer=kwiiyatta.analyze_wav)

    if conf.result_dir is None:
        result_path = source_path.with_suffix('.resynth.wav')
    else:
        result_path = pathlib.Path(conf.result_dir)/source_path.name

    feature = kwiiyatta.feature(source)

    wav = feature.synthesize()

    result_path.parent.mkdir(parents=True, exist_ok=True)
    wav.save(result_path)


if __name__ == '__main__':
    main()