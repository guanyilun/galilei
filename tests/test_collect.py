import os
import pickle

import numpy as np

from galilei import emulate
from galilei.experimental import collect
from galilei.sampling import build_samples


def test_collect():
    @collect(samples=build_samples({"a": [0, 2], "b": [0, 2]}, 100), save="_test.pkl")
    def test_em(a, b):
        x = np.linspace(0, 10, 100)
        return np.sin(a * x) + np.sin(b * x)

    @emulate(collection="_test.pkl", backend="sklearn")
    def test_em(a, b):
        pass

    # if collection loaded correctly, this should work
    output = test_em(1, 1)
    assert output.shape == (100,)

    # clean up
    if os.path.exists("_test.pkl"):
        os.remove("_test.pkl")
