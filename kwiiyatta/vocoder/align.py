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
    _, path = fastdtw.fastdtw(feature.mel_cepstrum.data,
                              target.mel_cepstrum.data,
                              radius=1, dist=2)
    return feature[list(project_path_iter(path))]


def align_even(a, b):
    _, path = fastdtw.fastdtw(a.mel_cepstrum.data, b.mel_cepstrum.data,
                              radius=1, dist=2)
    path = np.array(path).T
    return a[path[0]], b[path[1]]
