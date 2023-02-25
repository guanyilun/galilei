#!/usr/bin/env python
"""Tests for `galilei.sample` package."""
# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from galilei import sampling


def test_build_latin_hypercube():
    # Test case 1: Test if the returned array is of the correct size
    n = 5
    m = 4
    result = sampling.build_latin_hypercube(n, m)
    assert result.shape == (n, m)

    # Test case 2: Test if all elements of the returned array are between 0 and 1
    assert np.all(result >= 0) and np.all(result <= 1)

    # Test case 3: Test if all columns of the returned array are unique
    for i in range(m):
        assert len(set(result[:, i])) == n


def test_build_samples():
    # Sample data
    sample_dict = {"x1": [0, 1], "x2": [-1, 2], "x3": [5, 10]}
    nsamps = 10

    # Test case 1: Test if the returned dictionary has the correct number of keys
    result = sampling.build_samples(sample_dict, nsamps)
    assert len(result.keys()) == len(sample_dict.keys())

    # Test case 2: Test if the returned dictionary has the correct number of values in each key
    for key in result.keys():
        assert len(result[key]) == nsamps

    # Test case 3: Test if the returned dictionary values are within the correct range for each key
    for key in sample_dict.keys():
        assert np.all(result[key] >= sample_dict[key][0]) and np.all(
            result[key] <= sample_dict[key][1]
        )

    # Test case 4: Test if ValueError is raised when sample_dict values are not of length 2
    with pytest.raises(ValueError):
        sample_dict = {"x1": [0, 1, 2], "x2": [-1, 2], "x3": [5, 10]}
        sampling.build_samples(sample_dict, nsamps)

    # Test case 5: Test if AssertionError is raised when min is greater than or equal to max
    with pytest.raises(AssertionError):
        sample_dict = {"x1": [1, 0], "x2": [-1, 2], "x3": [5, 10]}
        sampling.build_samples(sample_dict, nsamps)

    # Test case 6: Test if AssertionError is raised when nsamps is less than or equal to 0
    with pytest.raises(AssertionError):
        sampling.build_samples(sample_dict, 0)
