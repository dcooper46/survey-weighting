"""
utility functions that work on dataframes
"""
from itertools import product

import pandas as pd
import numpy as np


def make_df(data, index, columns):
    """ create long dataframe from table like nd-array """
    ret_df = pd.DataFrame(np.squeeze(data), index=index, columns=columns)
    return ret_df.unstack().reset_index()


def group_counts(data, groups):
    """
    calculate group cell counts
    """
    counts = data.groupby(groups).size().reset_index()
    counts.columns.values[-1] = "n"
    return counts


def check_seed(seed, labels):
    """
    check if given raw data records or seed matrix
    """
    if isinstance(seed, pd.DataFrame):
        return records_to_seed(seed, labels)
    elif isinstance(seed, np.ndarray):
        return seed, seed.ravel()
    else:
        raise ValueError("error! malformed seed! check data.")


def records_to_seed(data, labels):
    """
    convert raw records to seed count matrix in shape
    determined by feature values
    """
    counts_full = get_full_counts(data, labels)
    seed_shape = tuple(len(val) for val in labels)
    return counts_full["n"].values.reshape(seed_shape), counts_full


def get_full_counts(data, unique_vals):
    """
    get counts of all possible feature values
     - unique_vals are lists of unique values for the demos being
       modeled, in the same order of the marginals provided
    """
    groupby_cols = data.columns.values.tolist()
    counts = group_counts(data, groupby_cols)
    counts.columns.values[-1] = "n"
    all_combos = pd.DataFrame(
        [p for p in product(*unique_vals)],
        columns=groupby_cols
    )
    return all_combos.merge(counts, how="left").fillna(0)
