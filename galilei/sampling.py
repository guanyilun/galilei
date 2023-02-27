from collections import OrderedDict

import numpy as np


def build_samples(sample_dict, nsamps, method="lhs"):
    """Build samples from a sample dictionary.

    Parameters
    ----------
    sample_dict : dict
        Dictionary of samples like {key: [min, max]}
    nsamps : int
        Number of samples to build.
    method : str, optional
        Method to use for building samples. The default is 'lhs'.

    Returns
    -------
    samples : dict
        Dictionary of samples.

    """
    # validation
    for key, value in sample_dict.items():
        if len(value) != 2:
            raise ValueError(
                "Sample dictionary values must be of length 2, for min and max!"
            )
        assert value[0] < value[1], "Min must be less than max!"
    assert nsamps > 0, "Number of samples must be greater than 0!"

    # build samples
    if method == "lhs":
        design_func = build_latin_hypercube
    else:
        return NotImplementedError("Only Latin hypercube sampling is implemented!")
    nparams = len(sample_dict)
    design_matrix = design_func(
        nsamps, nparams
    )  # n x m matrix with each element from 0 to 1

    # rescale design matrix to match the desired range given as min and max
    samples = OrderedDict()
    for i, key in enumerate(sample_dict.keys()):
        samples[key] = sample_dict[key][0] + design_matrix[:, i] * (
            sample_dict[key][1] - sample_dict[key][0]
        )
    return samples


# utility functions
def build_latin_hypercube(n, m):
    """Build a Latin hypercube of size n x m.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int
        Number of columns.

    Returns
    -------
    latin_hypercube : array_like
        Latin hypercube of size n x m.

    """
    latin_hypercube = np.zeros((n, m))
    for i in range(m):
        latin_hypercube[:, i] = np.random.permutation(n)
    design_matrix = np.zeros((n, m))
    for i in range(m):
        design_matrix[:, i] = (latin_hypercube[:, i] + np.random.uniform(size=n)) / n
    return design_matrix
