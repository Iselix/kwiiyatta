import copy
import pathlib

import kwiiyatta


def main():
    conf = kwiiyatta.Config()
    conf.add_argument('source', type=str,
                      help='Source wav file of voice resynthesis')
    conf.add_argument('--result-dir', type=str,
                      help='Path to write result wav files')
    conf.add_argument('--mcep', action='store_true',
                      help='Use mel-cepstrum to resynthesize')
    conf.add_argument('--play', action='store_true',
                      help='Play result wavform')
    conf.add_argument('--no-save', action='store_true',
                      help='Not to write result wav file, and play wavform')
    conf.add_argument('--carrier', type=str,
                      help='Wav file to use for carrier')
    conf.add_argument('--diffvc', action='store_true',
                      help='Use difference MelCepstrum synthesis')
    conf.add_argument('--result-fs', type=int,
                      help='Result waveform sampling rate')
    conf.parse_args()
    conf.play |= conf.no_save

    source_path = pathlib.Path(conf.source).resolve()
    source = conf.create_analyzer(source_path, Analyzer=kwiiyatta.analyze_wav)

    if conf.result_dir is None:
        result_path = source_path.with_suffix('.resynth.wav')
    else:
        result_path = pathlib.Path(conf.result_dir)/source_path.name

    feature = kwiiyatta.feature(source)

    wav = None

    if conf.carrier is not None:
        carrier = conf.create_analyzer(conf.carrier,
                                       Analyzer=kwiiyatta.analyze_wav)
        feature = kwiiyatta.align(source, carrier, pad_silence=False)
        if conf.diffvc:
            mcep_diff = copy.copy(feature.mel_cepstrum)
            mcep_diff.data -= carrier.mel_cepstrum.data
            wav = kwiiyatta.apply_mlsa_filter(carrier.wavdata, mcep_diff)
        else:
            feature.f0 = carrier.f0

    if wav is None:
        if conf.mcep:
            feature.extract_mel_cepstrum()
            feature.spectrum_envelope = None
        if conf.result_fs is not None:
            feature.resample(conf.result_fs)
        wav = feature.synthesize()

    if not conf.no_save:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        wav.save(result_path)

    if conf.play:
        wav.play()


if __name__ == '__main__':
    main()
