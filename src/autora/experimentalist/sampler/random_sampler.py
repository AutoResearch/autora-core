import random
from functools import singledispatch
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from autora.utils.deprecation import deprecated_alias


@singledispatch
def random_sample(conditions, num_samples: int = 1, random_state: Optional[int] = None):
    """
    Uniform random sampling without replacement from a pool of conditions.

    Args:
        conditions: Pool of conditions
        num_samples: number of samples to collect
        random_state: initialization parameter for the random sampler

    Returns: Sampled pool

    """

    raise NotImplementedError("%s is not supported" % (type(conditions)))


@random_sample.register(list)
@random_sample.register(range)
@random_sample.register(filter)
def random_sample_sequence(
    conditions, num_samples: int = 1, random_state: Optional[int] = None
):
    """
    Single dispatch variant of random_sample for list, range and filter objects

    Examples:
        From a range:
        >>> random.seed(1)
        >>> random_sample(range(100), num_samples=5)
        [53, 37, 65, 51, 4]

        >>> random.seed(1)
        >>> random_sample([1,2,3,4,5,6,7,8,9,10], num_samples=5)
        [7, 9, 10, 8, 6]

        >>> random.seed(1)
        >>> random_sample(filter(lambda x: (x % 3 == 0) & (x % 5 == 0), range(1_000)),
        ...     num_samples=5)
        [375, 390, 600, 285, 885]

    """
    if random_state is not None:
        random.seed(random_state)
    if isinstance(conditions, Iterable):
        conditions = list(conditions)
    random.shuffle(conditions)
    samples = conditions[0:num_samples]

    return samples


@random_sample.register(pd.DataFrame)
def random_sample_dataframe(
    conditions, num_samples: int = 1, random_state: Optional[int] = None
):
    """
    Single dispatch variant of random_sample for pd.DataFrames

    Examples:
        From a pd.DataFrame:
        >>> import pandas as pd
        >>> random.seed(1)
        >>> random_sample(pd.DataFrame({"x": range(100, 200)}), num_samples=5, random_state=180)
              x
        67  167
        71  171
        64  164
        63  163
        96  196

    """
    return pd.DataFrame.sample(
        conditions, random_state=random_state, n=num_samples, replace=False
    )


@random_sample.register(np.ndarray)
@random_sample.register(np.recarray)
def random_sample_np_array(
    conditions, num_samples: int = 1, random_state: Optional[int] = None
):
    """
    Single dispatch variant of random_sample for np.ndarray and np.recarray

    Examples:
        From a pd.DataFrame:
        >>> import numpy as np
        >>> random_sample(np.linspace([-1, -100], [1, 100], 101), num_samples=5,
        ...     random_state=180)
        array([[  0.12,  12.  ],
               [ -0.18, -18.  ],
               [ -0.54, -54.  ],
               [  0.32,  32.  ],
               [  0.96,  96.  ]])

        >>> random_sample(np.core.records.fromrecords([
        ...     ("a", 1, "alpha"),
        ...     ("b", 2, "beta"),
        ...     ("c", 3, "gamma"),
        ...     ("d", 4, "delta"),
        ... ], names=["l", "n", "g"]), num_samples=2, random_state=1)
        array([('b', 2, 'beta'), ('c', 3, 'gamma')],
              dtype=(numpy.record, [('l', '<U1'), ('n', '<i...'), ('g', '<U5')]))
    """
    rng = np.random.default_rng(random_state)
    return rng.choice(conditions, size=num_samples, replace=False)


random_sampler = deprecated_alias(random_sample, "random_sampler")
