import fastdtw

import numpy as np


def project_path_iter(path):
    prev_y = -1
    for x, y in path:
        if y == prev_y:
            continue
        yield x
        prev_y = y


def align(feature, target):
    fs = min(feature.fs, target.fs)
    _, path = fastdtw.fastdtw(feature.resample_mel_cepstrum(fs).data,
                              target.resample_mel_cepstrum(fs).data,
                              radius=1, dist=2)
    return feature[list(project_path_iter(path))]


def align_even(a, b):
    fs = min(a.fs, b.fs)
    _, path = fastdtw.fastdtw(a.resample_mel_cepstrum(fs).data,
                              b.resample_mel_cepstrum(fs).data,
                              radius=1, dist=2)
    path = np.array(path).T
    return a[path[0]], b[path[1]]
