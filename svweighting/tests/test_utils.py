""" tests for weighting utils """
from __future__ import absolute_import

from emarkets.utils.utils import (
    Marginal, nested_list, make_df, build_slice_indices, safe_divide
)
import numpy as np


def test_nested_list():
    nested = [["a"], ["b", "c"], "d"]
    not_nested = [1, 2, 3]
    assert nested_list(nested)
    assert not nested_list(not_nested)


def test_add_lists():
    list_a = ["a", "b", "c"]
    list_b = ["d", "e", "f"]
    list_c = [["g", "h"], ["i", "j"]]
    list_ab = [["a", "b", "c"], ["d", "e", "f"]]
    list_ac = [["a", "b", "c"], ["g", "h"], ["i", "j"]]
    assert Marginal.add_lists(list_a, list_b) == list_ab
    assert Marginal.add_lists(list_a, list_c) == list_ac


def test_outer_vec():
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    expected = np.array([
        [4, 8, 12],
        [5, 10, 15],
        [6, 12, 18]
    ])
    assert np.allclose(Marginal.outer(vec1, vec2), expected)


def test_outer_mat():
    vec = np.array([1, 2])
    mat = np.ones((2, 2))
    expected = np.array([
        [[1, 1], [1, 1]],
        [[2, 2], [2, 2]]
    ])
    assert np.allclose(Marginal.outer(mat, vec), expected)


def test_slice_build():
    expected = [np.newaxis, slice(None), slice(None), np.newaxis]
    assert build_slice_indices(4, (1, 2)) == expected


def test_safe_divide():
    arr1 = np.array([1., 0., 2., 3.])
    arr2 = np.array([1.5, 2., 0., 1.3])
    expected = np.array([0.667, 0., 0., 2.308])
    assert np.allclose(safe_divide(arr1, arr2), expected, atol=0.001)
