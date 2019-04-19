from nnmnkwii.baseline.gmm import MLPG

from sklearn.mixture import GaussianMixture

from . import abc, delta


class GMMFeatureConverter(abc.FeatureConverter):
    def __init__(self, components=64, max_iter=100, random_state=None,
                 **kwargs):
        super().__init__()
        self.init_gmm(components, max_iter, random_state, **kwargs)

    def init_gmm(self, components, max_iter=100, random_state=None, **kwargs):
        if 'verbose' not in kwargs:
            kwargs['verbose'] = 1
        if 'covariance_type' not in kwargs:
            kwargs['covariance_type'] = 'full'

        self.gmm = GaussianMixture(
            n_components=components, max_iter=max_iter,
            random_state=random_state, **kwargs
        )

    def _train(self, dataarray, **kwargs):
        self.gmm.fit(dataarray, **kwargs)

    def convert(self, feature, mlpg=True, diff=False):
        windows = delta.DELTA_WINDOWS
        if not mlpg:
            windows = windows[0:1]
        paramgen = MLPG(self.gmm, windows=windows, diff=diff)

        return paramgen.transform(feature)
