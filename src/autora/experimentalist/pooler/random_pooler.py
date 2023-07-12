import random
from typing import Iterable, List, Tuple

import numpy as np

from autora.utils.deprecation import deprecated_alias
from autora.variable import IV


def random_pool(
    ivs: List[IV], num_samples: int = 1, duplicates: bool = True
) -> Iterable:
    """
    Creates combinations from lists of discrete values using random selection.
    Args:
        ivs: List of independent variables
        n: Number of samples to sample
        duplicates: Boolean if duplicate value are allowed.

    """
    l_samples: List[Tuple] = []
    # Create list of pools of values sample from
    l_iv_values = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"gridsearch_pool only supports independent variables with discrete allowed values, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)

    # Check to ensure infinite search won't occur if duplicates not allowed
    if not duplicates:
        l_pool_len = [len(set(s)) for s in l_iv_values]
        n_combinations = np.product(l_pool_len)
        try:
            assert num_samples <= n_combinations
        except AssertionError:
            raise AssertionError(
                f"Number to sample n({num_samples}) is larger than the number "
                f"of unique combinations({n_combinations})."
            )

    # Random sample from the pools until n is met
    while len(l_samples) < num_samples:
        l_samples.append(tuple(map(random.choice, l_iv_values)))
        if not duplicates:
            l_samples = [*set(l_samples)]

    return iter(l_samples)


random_pooler = deprecated_alias(random_pool, "random_pooler")
