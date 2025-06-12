import pytest
from readysetgo.utils.common_functions import flatten_upper_triangle
import numpy as np


def test_flatten_upper_triangle_basic():
    matrix = np.array([
        [1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]
    ])
    result = flatten_upper_triangle(matrix)
    expected = np.array([1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(result, expected)

def test_all_zeros():
    matrix = np.zeros((3, 3))
    result = flatten_upper_triangle(matrix)
    expected = np.array([])  # No non-zero values
    np.testing.assert_array_equal(result, expected)

def test_non_square_matrix():
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    with pytest.raises(ValueError):
        flatten_upper_triangle(matrix)

def test_negative_and_zero_values():
    matrix = np.array([
        [0, -1, 2],
        [0, 0, 0],
        [0, 0, -3]
    ])
    result = flatten_upper_triangle(matrix)
    expected = np.array([2])  # Only positive non-zero values
    np.testing.assert_array_equal(result, expected)

def test_diagonal_only():
    matrix = np.diag([1, 2, 3])
    result = flatten_upper_triangle(matrix)
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)