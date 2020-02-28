import itertools

import fastdtw

import numpy as np

import kwiiyatta


def binalize(x, threshold, ceil, floor=0, out=None):
    if out is None:
        out = np.full_like(x, floor)
    else:
        out[:] = floor

    out[x >= threshold] = ceil
    return out


def make_feature(f, fs, vuv='voiced', vuv_weight=9.0,
                 power='binalize', power_weight=9.4,
                 power_pivot='max', power_threshold=1.636):
    data = f.resample_mel_cepstrum(fs).data
    data_power = data[:, 0]

    feature = np.hstack((np.zeros((len(data), 2)), data[:, 1:]))

    if power == 'binalize':
        if power_pivot == 'max':
            threshold = data_power.max() - power_threshold
        elif power_pivot == 'median':
            threshold = np.median(data_power) - power_threshold
        elif power_pivot == 'min':
            threshold = data_power.min() + power_threshold
        elif power_pivot == 'fix':
            threshold = power_threshold
        else:
            raise ValueError(f'Unknown power_pivot parameter: {power_pivot!r}')

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


def dtw_feature(x, y, vuv='voiced', power='binalize', strict=True,
                radius=32, **kwargs):
    fs = min(x.fs, y.fs)

    kwargs['vuv'] = vuv
    kwargs['power'] = power

    x_feature = make_feature(x, fs, **kwargs)
    y_feature = make_feature(y, fs, **kwargs)

    dist, path = fastdtw.fastdtw(x_feature, y_feature, dist=2, radius=radius)

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


def project_path_iter(path, trim=True, trim_len=1):
    prev_x = prev_y = -1
    if trim:
        prev_y += trim_len
    len_y = path[-1][1] + 1
    if trim:
        len_y -= trim_len
    for x, y in path:
        if y <= prev_y:
            continue
        elif y - prev_y > 1:
            y = min(y, len_y-1)
            diff_x = x - prev_x
            diff_y = y - prev_y
            for i in range(diff_y):
                yield prev_x + diff_x * i // (diff_y-1)
        elif y >= len_y:
            break
        else:
            yield x
        prev_x = x
        prev_y = y


def align(feature, target, vuv='f0', strict=False,
          pad_silence=True, pad_len=100,
          **kwargs):
    if pad_silence:
        feature = kwiiyatta.pad_silence(feature, frame_len=pad_len)
        target = kwiiyatta.pad_silence(target, frame_len=pad_len)
    _, path = dtw_feature(feature, target, vuv=vuv, strict=strict, **kwargs)
    return feature[list(project_path_iter(path, trim=pad_silence,
                                          trim_len=pad_len))]


def align_even(a, b, pad_silence=True, pad_len=100, **kwargs):
    if pad_silence:
        a = kwiiyatta.pad_silence(a, pad_len)
        b = kwiiyatta.pad_silence(b, pad_len)
    _, path = dtw_feature(a, b, **kwargs)
    path = np.array(path).T
    if pad_silence:
        begin = np.argmax(np.logical_and(path[0] >= pad_len,
                                         path[1] >= pad_len))
        end = np.argmax(np.logical_and(path[0] >= a.frame_len-pad_len,
                                       path[1] >= b.frame_len-pad_len))
        path = path[:, begin:end]
    return a[path[0]], b[path[1]]
