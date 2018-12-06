import pathlib

import kwiiyatta


def main():
    conf = kwiiyatta.Config()
    conf.add_argument('source', type=str,
                      help='Source wav file of voice resynthesis')
    conf.add_argument('--result-dir', type=str,
                      help='Path to write result wav files')
    conf.add_argument('--play', action='store_true',
                      help='Play result wavform')
    conf.add_argument('--no-save', action='store_true',
                      help='Not to write result wav file, and play wavform')
    conf.parse_args()
    conf.play |= conf.no_save

    source_path = pathlib.Path(conf.source).resolve()
    source = conf.create_analyzer(source_path, Analyzer=kwiiyatta.analyze_wav)

    if conf.result_dir is None:
        result_path = source_path.with_suffix('.resynth.wav')
    else:
        result_path = pathlib.Path(conf.result_dir)/source_path.name

    feature = kwiiyatta.feature(source)

    wav = feature.synthesize()

    if not conf.no_save:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        wav.save(result_path)

    if conf.play:
        wav.play()


if __name__ == '__main__':
    main()
