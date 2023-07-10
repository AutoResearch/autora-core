import random
from typing import Iterable, Optional

import pandas as pd

from autora.state.delta import Result, wrap_to_use_state
from autora.utils.deprecation import deprecated_alias


def random_sample(conditions, num_samples: int = 1, random_state: Optional[int] = None):
    """


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


random_sampler = deprecated_alias(random_sample, "random_sampler")


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


random_sample_executor = wrap_to_use_state(random_sample_from_conditions)
random_sample_executor.__doc__ = """
Examples:
    >>> from autora.state.delta import State
    >>> from autora.variable import VariableCollection, Variable
    >>> from dataclasses import dataclass, field
    >>> import pandas as pd
    >>> import numpy as np
    >>> from autora.experimentalist.pooler.grid import grid_pool_executor

    We define a state object with the fields we need:
    >>> @dataclass(frozen=True)
    ... class S(State):
    ...     variables: VariableCollection = field(default_factory=VariableCollection)
    ...     conditions: pd.DataFrame = field(default_factory=pd.DataFrame,
    ...                                      metadata={"delta": "replace"})

    With one independent variable "x", and some allowed values:
    >>> s = S(
    ...     variables=VariableCollection(independent_variables=[
    ...         Variable(name="x", allowed_values=range(100))
    ... ]))

    ... we can update the state with a sample from the allowed values:
    >>> s_ = grid_pool_executor(s)
    >>> random_sample_executor(s_, num_samples=5, random_state=1
    ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    S(variables=..., conditions=     x
    80  80
    84  84
    33  33
    81  81
    93  93)

"""
