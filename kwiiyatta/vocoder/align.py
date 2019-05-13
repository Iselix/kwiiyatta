import itertools

import fastdtw

import numpy as np


def binalize(x, threshold, ceil, floor=0, out=None):
    if out is None:
        out = np.full_like(x, floor)
    else:
        out[:] = floor

    out[x >= threshold] = ceil
    return out


def make_feature(f, fs, vuv='voiced', vuv_weight=8,
                 power='binalize', power_weight=16):
    data = f.resample_mel_cepstrum(fs).data
    data_power = data[:, 0]

    feature = np.hstack((np.zeros((len(data), 2)), data[:, 1:]))

    if power == 'binalize':
        threshold = data_power.min() + 2
        binalize(data_power, threshold, power_weight,
                 out=feature[:, 0])
    elif power == 'raw':
        feature[:, 0] = data_power
    elif power is None:
        pass
    else:
        raise ValueError(f'Unknown power parameter: {power!r}')

    if vuv == 'voiced':
        feature[:, 1][f.is_voiced] = vuv_weight
    elif vuv == 'f0':
        feature[:, 1][f.f0 > 0] = vuv_weight
    elif vuv is None:
        pass
    else:
        raise ValueError(f'Unknown vuv parameter: {vuv!r}')

    return feature


def dtw_feature(x, y, vuv='voiced', power='binalize', strict=True, **kwargs):
    fs = min(x.fs, y.fs)

    kwargs['vuv'] = vuv
    kwargs['power'] = power

    x_feature = make_feature(x, fs, **kwargs)
    y_feature = make_feature(y, fs, **kwargs)

    dist, path = fastdtw.fastdtw(x_feature, y_feature, dist=2)

    def check(x, y):
        if power == 'binalize':
            if (x_feature[x, 0] > 0) ^ (y_feature[y, 0] > 0):
                return False
        if vuv is not None:
            if (x_feature[x, 1] > 0) ^ (y_feature[y, 0] > 0):
                return False
        return True

    if strict:
        path = np.fromiter(
                itertools.chain(
                    path[0],
                    itertools.chain.from_iterable(
                        [x, y] for x, y in path[1:-1] if check(x, y)
                    ),
                    path[-1]
                ),
                np.int
        ).reshape((-1, 2))
    else:
        path = np.array(path)

    return dist, path


def project_path_iter(path):
    prev_x = prev_y = -1
    for x, y in path:
        if y == prev_y:
            continue
        if y - prev_y > 1:
            diff_x = x - prev_x
            diff_y = y - prev_y
            for i in range(diff_y):
                yield prev_x + diff_x * i // (diff_y-1)
        else:
            yield x
        prev_x = x
        prev_y = y


def align(feature, target, vuv='f0', strict=False, **kwargs):
    _, path = dtw_feature(feature, target, vuv=vuv, strict=strict, **kwargs)
    return feature[list(project_path_iter(path))]


def align_even(a, b, **kwargs):
    _, path = dtw_feature(a, b, **kwargs)
    path = np.array(path).T
    return a[path[0]], b[path[1]]
