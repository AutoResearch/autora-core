import random
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

from autora.state.delta import Result, wrap_to_use_state
from autora.utils.deprecation import deprecated_alias


def random_sample_from_conditions_iterable(
    conditions: Union[Iterable, Sequence], num_samples: int = 1
):
    """
    Uniform random sampling without replacement from a pool of conditions.
    Args:
        conditions: Pool of conditions
        num_samples: number of samples to collect

    Returns: Sampled pool

    Examples:
        From a range:
        >>> random.seed(1)
        >>> random_sample_from_conditions_iterable(range(100), num_samples=5)
        [53, 37, 65, 51, 4]

        >>> random.seed(1)
        >>> random_sample_from_conditions_iterable([1,2,3,4,5,6,7,8,9,10], num_samples=5)
        [7, 9, 10, 8, 6]

        >>> random.seed(1)
        >>> random_sample_from_conditions_iterable(
        ...     filter(lambda x: (x % 3 == 0) & (x % 5 == 0), range(1_000)),
        ...     num_samples=5
        ... )
        [375, 390, 600, 285, 885]

    """
    if isinstance(conditions, Iterable):
        conditions = list(conditions)
    random.shuffle(conditions)
    samples = conditions[0:num_samples]

    return samples


random_sampler = deprecated_alias(
    random_sample_from_conditions_iterable, "random_sampler"
)


def random_sample_from_conditions(
    conditions,
    num_samples: int = 1,
    random_state: Optional[int] = None,
    replace: bool = False,
) -> Result:
    """
    Take a random sample from some conditions.

    Args:
        conditions: the conditions to sample from
        num_samples:
        random_state:
        replace:

    Returns: a Result object with a field `conditions` with a DataFrame of the sampled conditions

    Examples:
        From a pd.DataFrame:
        >>> import pandas as pd
        >>> random.seed(1)
        >>> random_sample_from_conditions(
        ...     pd.DataFrame({"x": range(100, 200)}), num_samples=5, random_state=180)
        {'conditions':       x
        67  167
        71  171
        64  164
        63  163
        96  196}

    """
    return Result(
        conditions=pd.DataFrame.sample(
            conditions, random_state=random_state, n=num_samples, replace=replace
        )
    )


random_sample = wrap_to_use_state(random_sample_from_conditions)
