# Iterative proportional fitting
from __future__ import division, absolute_import

import copy
import numpy as np

from ..utils import build_slice_indices, safe_divide
from ..df_utils import check_seed


class IPF():
    """
    documentation
    """
    def __init__(self, marginals, marginal_dims,
                 max_iter=50, error_threshold=0.005, error_rate=1e-5):
        self.marginals = marginals
        self.marginal_dims = marginal_dims
        self.max_iter = max_iter
        self.error_threshold = error_threshold
        self.error_rate = error_rate
        self.converged = False
        self.weights = None
        self.scaled_weights = None
        self.counts = None

    # TODO: add check to ensure marginals have same pop - else use proportions

    # TODO: add evaluation / goodness-of-fit methods

    def fit(self, data, labels=None):
        """
        run iterative proportional fitting procedure

        Parameters
        ----------
        data: pandas DataFrame of raw records or numpy matrix of seed counts
        marginals: list of marginal totals
        marginal_dims: list of dimensions for corresponding marginals
            note: these match the corresponding features in the data which
                  were used to create the marginal. if a dataframe is
                  passed, these should be the column index that produced
                  the tabular count
        Returns
        -------
        self : object
            Returns self.
        """
        seed, self.counts = check_seed(data, labels)

        result = copy.copy(seed)
        i = 0
        error = 1
        while i <= self.max_iter:
            prev_result = result
            prev_error = error
            result = _ipfp_step(result, self.marginals, self.marginal_dims)
            error = np.max(np.abs(result - prev_result))
            stop = (error < self.error_threshold or
                    abs(error - prev_error) < self.error_rate)
            if stop:
                break
            i += 1
        if i < self.max_iter:
            self.converged = True
        self.weights = result
        self.scaled_weights = result / seed
        self.iter = i

        return self

    def transform(self, data):
        """
        transform raw data/df by assigning weights based on fitted
        cross-tab totals
        """
        return_counts = self.counts.drop("n", axis=1)
        return_counts["wgt"] = self.scaled_weights.ravel()
        return data.merge(return_counts, how="left")

    def fit_transform(self, data, labels=None, model_cols=None):
        """
        Fit weights via ipfp and add/update weights in data
        """
        if model_cols is not None:
            fit_data = data[model_cols]
        else:
            fit_data = data
        self.fit(fit_data, labels)
        return self.transform(data)


def _ipfp_step(seed, marginals, marginal_dims):
    """
    single step of the ipfp procedure that adjusts cell values in a seed
    along each margin with expected totals/percents
    """
    ndim = len(seed.shape)
    seed_dims = range(ndim)
    for (marg_dim, marg) in zip(marginal_dims, marginals):
        sum_dims = tuple(d for d in seed_dims if d not in marg_dim)
        seed_sum = seed.sum(sum_dims)
        scale = safe_divide(marg, seed_sum)  # or other checks to avoid div / 0
        scale_indices = build_slice_indices(ndim, marg_dim)
        seed = seed * scale[scale_indices]
    return seed
