""" utility functions and classes for weighting """
from itertools import product

import numpy as np
import pandas as pd

# should never need more than this
AXIS_LETTERS = 'ijklmnopqr'


class Marginal():
    """
    container for a marginal (raw or calculated)
    """
    def __init__(self, name=None, labels=None, values=None):
        self.name = name
        self.labels = labels
        self.values = values
        self.is_empty = self.labels is None or self.values is None

    def __mul__(self, other):
        values = self.outer(self.values, other.values)
        labels = self.add_lists(self.labels, other.labels)
        name = "|".join([self.name, other.name])
        return Marginal(name, labels, values)

    def __nonzero__(self):
        return self.is_empty

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def add_lists(list_a, list_b):
        """
        given two lists, return new lists containing both;
        if one or both are nested, append
        """
        a_nested = nested_list(list_a)
        b_nested = nested_list(list_b)

        if a_nested and not b_nested:
            return list_a + [list_b]
        elif b_nested and not a_nested:
            return [list_a] + list_b
        elif not a_nested and not b_nested:
            return [list_a] + [list_b]
        else:
            return list_a + list_b

    @staticmethod
    def outer(array, vector, axis_letters=AXIS_LETTERS):
        """
        compute outer product of an arbitrary array (1d or nd) and a vector
        using einstein notation
        """
        array_subs = axis_letters[:array.ndim]
        vec_sub = axis_letters[array.ndim]
        action = '{aa},{ba} -> {ba}{aa}'.format(aa=array_subs, ba=vec_sub)
        return np.einsum(action, array, vector)


def nested_list(check_list):
    """ check if a list is nested (list of lists) """
    return any(isinstance(i, list) for i in check_list)


def mvmult(mat, vec):
    """ multiply a matrix by each element in a vector """
    return np.array([mat * vi for vi in vec])


def build_slice_indices(ndim, dims):
    """
    create index array where only certain indices are known
    """
    ret = [np.newaxis for _ in range(ndim)]
    for dim in dims:
        ret[dim] = slice(None)
    return ret


def safe_divide2(arr1, arr2):
    """
    safely divide arrays by checking for 0
    """
    if isinstance(arr1, np.ndarray) or isinstance(arr2, np.ndarray) or \
            isinstance(arr1, pd.Series) or isinstance(arr2, pd.Series):
        return np.array([safe_divide(x, y) for x, y in zip(arr1, arr2)])
    elif isinstance(arr1, list) or isinstance(arr2, list):
        return safe_divide(np.array(arr1), np.array(arr2))
    else:
        return arr1 / arr2 if arr2 != 0 else 0


def safe_divide(arr1, arr2):
    """
    safely divide arrays by checking for 0
    """
    return np.divide(arr1, arr2, out=np.zeros_like(arr1), where=arr2 != 0)


def calculate_marginals2d(expected_counts, labels, label_maps):
    """
    calculate expected marginal cell proportions
        - assume no greater than 2dim cross tabs
    keys of dict should tuples
    """
    marginals = []
    marginal_dims = []
    for label, value_counts in expected_counts.items():
        if not isinstance(label, (tuple, list)):
            label = tuple([label])
        if len(label) == 1:
            marginals.append([value_counts[l] for l in label_maps[label[0]]])
            marginal_dims.append([labels.index(label[0])])
        else:
            shape = tuple(len(label_maps[l]) for l in label)
            sub_marginal = np.zeros(shape)
            ids = list(product(*[enumerate(label_maps[j]) for j in label]))
            for d1, d2 in ids:
                i, label_i = d1
                j, label_j = d2
                sub_marginal[i][j] = value_counts[label_i][label_j]
            marginals.append(sub_marginal.tolist())
            marginal_dims.append([labels.index(d) for d in label])
    return marginals, marginal_dims
