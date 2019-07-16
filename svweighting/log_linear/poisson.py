"""
Model respondent weights as Poisson log-linear
"""
from __future__ import division, absolute_import

from itertools import product
from functools import reduce

import operator
import pandas as pd
import numpy as np
import statsmodels.api as sm

from ..utils import Marginal
from ..df_utils import group_counts, make_df


class PoissonModel():
    """
    predict respondent level weights using poisson log-linear model
    """
    def __init__(self, demo_totals, normalize=True):
        self.demo_totals = demo_totals
        self.demos = demo_totals.keys()
        self.population = sum(demo_totals.values()[0].values())
        self.normalize = normalize
        self.marginals = Marginal()
        self.model = None
        self.actual_counts = None
    # TODO: add check to ensure marginals have same pop - else use proportions

    def calculate_marginals(self):
        """
        combine individual marginals into
        cross-tabs of expected counts/fractions
        """
        raw_marginals = []
        for demo, demo_map in self.demo_totals.items():
            name = demo
            labels = demo_map.keys()
            values = np.array([demo_map[l] for l in labels])
            if self.normalize:
                values /= self.population
            raw_marginals.append(Marginal(name, labels, values))
        reduced = reduce(operator.mul, raw_marginals)
        self.marginals = reduced

    def fit(self, data):
        """
        fit log-linear model to predicted weights
        """
        if self.marginals.is_empty:
            self.calculate_marginals()
        self.actual_counts = group_counts(data, self.demos)
        expected_counts = _get_expected_values(
            self.marginals, self.demos, self.population
        )

        model_data = sm.add_constant(
            pd.get_dummies(
                pd.merge(self.actual_counts, expected_counts,),
                columns=self.demos
            ),
            prepend=False
        )
        features = model_data.columns[2:]
        model = sm.GLM(
            model_data["N"],
            model_data[features],
            family=sm.families.Poisson(),
            offset=np.log(model_data["n"])
        ).fit()
        self.model = model
        return self

    def predict(self, data, keep_all=True):
        """
        predict using fitted values from model to get
        expected population values as weights
        - scale them by initial counts so the raw records sum
          to the correct totals and marginals
        """
        self.actual_counts["wgts"] = self.model.predict()
        self.actual_counts["scaled_wgts"] = (
            self.actual_counts["wgts"] / self.actual_counts["n"]
        )
        if keep_all:
            keep = self.actual_counts.columns
        else:
            keep = self.demos + ["scaled_wgts"]
        return pd.merge(data, self.actual_counts[keep], how="left")


def _get_expected_values(marginals, demos, market_pop):
    """
    expand calculated marginal values into expected
    values by demo label
    """
    columns = marginals.labels[0]
    index = marginals.labels[1]
    if len(marginals) > 2:
        splits, split_dims, split_labels = _get_splits(marginals)
        frames = [make_df(s, index, columns) for s in splits]
        label_vals = _get_labels(split_dims, split_labels)

        for i, frame in enumerate(frames):
            for j, label in enumerate(label_vals[i]):
                frame.insert(j + 2, "level_{}".format(j + 2), label)

        expected = pd.concat(frames).reset_index(drop=True)
    else:
        expected = make_df(
            marginals.values, index, columns
        ).reset_index(drop=True)

    expected.columns = demos + ["p"]
    expected["N"] = expected["p"] * market_pop
    expected.drop("p", axis=1, inplace=True)
    return expected


def _get_splits(marginals):
    """
    split marginal matrix along each axis
    """
    split_labels = marginals.labels[2:][::-1]  # reverse order of labels
    split_dims = marginals.values.shape[:-2]
    n_splits = len(split_dims)

    splits = [
        np.squeeze(f)
        for f in np.split(marginals.values, split_dims[0], 0)
    ]
    if n_splits > 1:
        for dim in split_dims[1:]:
            splits = [
                np.squeeze(f)
                for df in splits for f in np.split(df, dim, 0)
            ]
    return splits, split_dims, split_labels


def _get_labels(split_dims, split_labels):
    """
    get labels for marginal features
    """
    label_indexes = tuple(range(len(split_dims)))
    ids = [range(d) for d in split_dims]
    label_ids = list(product(*ids))
    label_lookup = [zip(label_indexes, lid) for lid in label_ids]
    label_vals = [
        [split_labels[j][k]
         for j, k in idxs] for idxs in label_lookup
    ]
    return label_vals
