"""Tools to make randomly sampled experimental conditions."""
import random
from functools import singledispatch
from typing import Optional, Union

import numpy as np
import pandas as pd

from autora.state.delta import Result, State, wrap_to_use_state
from autora.variable import ValueType, VariableCollection


@singledispatch
def random_pool(s, **kwargs):
    """Function to create a sequence of conditions randomly sampled from independent variables."""
    raise NotImplementedError(
        "random_pool doesn't have an implementation for %s (type=%s)" % (s, type(s))
    )


@random_pool.register(State)
def random_pool_on_state(
    s: State,
    num_samples: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
) -> State:
    """

    Args:
        variables:
        fmt: the output type required

    Returns:

    Examples:
        >>> from autora.state.delta import State
        >>> from autora.variable import VariableCollection, Variable
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> import numpy as np

        We define a state object with the fields we need:
        >>> @dataclass(frozen=True)
        ... class S(State):
        ...     variables: VariableCollection = field(default_factory=VariableCollection)
        ...     conditions: pd.DataFrame = field(default_factory=pd.DataFrame,
        ...                                      metadata={"delta": "replace"})

        With one independent variable "x", and some allowed_values:
        >>> s = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=range(10))
        ... ]))

        ... we get some of those values back when running the experimentalist:
        >>> random_pool(s, random_state=1).conditions
           x
        0  4
        1  5
        2  7
        3  9
        4  0

        With one independent variable "x", and a value_range:
        >>> t = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", value_range=(-5, 5))
        ... ]))

        ... we get a sample of the range back when running the experimentalist:
        >>> random_pool(t, random_state=1).conditions
                  x
        0  0.118216
        1  4.504637
        2 -3.558404
        3  4.486494
        4 -1.881685



        The allowed_values or value_range must be specified:
        >>> random_pool(
        ...     S(variables=VariableCollection(independent_variables=[Variable(name="x")])))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...

        With two independent variables, we get independent samples on both axes:
        >>> t = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=range(1, 5)),
        ...         Variable(name="x2", allowed_values=range(1, 500)),
        ... ]))
        >>> random_pool(t,
        ...                            num_samples=10, replace=True, random_state=1).conditions
           x1   x2
        0   2  434
        1   3  212
        2   4  137
        3   4  414
        4   1  129
        5   1  205
        6   4  322
        7   4  275
        8   1   43
        9   2   14

        If any of the variables have unspecified allowed_values, we get an error:
        >>> random_pool(S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2"),
        ... ])))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...


        We can specify arrays of allowed values:
        >>> u = S(
        ...     variables=VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
        ...         Variable(name="y", allowed_values=[3, 4]),
        ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
        ... ]))
        >>> random_pool(u, random_state=1).conditions
             x  y     z
        0 -0.6  3  29.0
        1  0.2  4  24.0
        2  5.2  4  23.0
        3  9.0  3  29.0
        4 -9.4  3  22.0
    """
    return wrap_to_use_state(random_pool_on_variables)(
        s, num_samples=num_samples, random_state=random_state, replace=replace
    )


@random_pool.register(VariableCollection)
def random_pool_on_variables(
    variables: VariableCollection,
    num_samples: int = 5,
    random_state: Optional[int] = None,
    replace: bool = True,
) -> pd.DataFrame:
    """

    Args:
        variables: the description of all the variables in the AER experiment.
        num_samples: the number of conditions to produce
        random_state: the seed value for the random number generator
        replace: if True, allow repeated values

    Returns: a Result / Delta object with the conditions as a pd.DataFrame in the `conditions` field

    Examples:
        >>> from autora.state.delta import State
        >>> from autora.variable import VariableCollection, Variable
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> import numpy as np

        With one independent variable "x", and some allowed_values we get some of those values
        back when running the experimentalist:
        >>> random_pool(
        ...     VariableCollection(
        ...         independent_variables=[Variable(name="x", allowed_values=range(10))
        ... ]), random_state=1)
        {'conditions':    x
        0  4
        1  5
        2  7
        3  9
        4  0}


        ... we get a sample of the range back when running the experimentalist:
        >>> random_pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x", value_range=(-5, 5))
        ... ]), random_state=1)["conditions"]
                  x
        0  0.118216
        1  4.504637
        2 -3.558404
        3  4.486494
        4 -1.881685



        The allowed_values or value_range must be specified:
        >>> random_pool(VariableCollection(independent_variables=[Variable(name="x")]))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...

        With two independent variables, we get independent samples on both axes:
        >>> random_pool(VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=range(1, 5)),
        ...         Variable(name="x2", allowed_values=range(1, 500)),
        ... ]), num_samples=10, replace=True, random_state=1)["conditions"]
           x1   x2
        0   2  434
        1   3  212
        2   4  137
        3   4  414
        4   1  129
        5   1  205
        6   4  322
        7   4  275
        8   1   43
        9   2   14

        If any of the variables have unspecified allowed_values, we get an error:
        >>> random_pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x1", allowed_values=[1, 2]),
        ...         Variable(name="x2"),
        ... ]))
        Traceback (most recent call last):
        ...
        ValueError: allowed_values or [value_range and type==REAL] needs to be set...


        We can specify arrays of allowed values:

        >>> random_pool(
        ...     VariableCollection(independent_variables=[
        ...         Variable(name="x", allowed_values=np.linspace(-10, 10, 101)),
        ...         Variable(name="y", allowed_values=[3, 4]),
        ...         Variable(name="z", allowed_values=np.linspace(20, 30, 11)),
        ... ]), random_state=1)["conditions"]
             x  y     z
        0 -0.6  3  29.0
        1  0.2  4  24.0
        2  5.2  4  23.0
        3  9.0  3  29.0
        4 -9.4  3  22.0


    """
    rng = np.random.default_rng(random_state)

    raw_conditions = {}
    for iv in variables.independent_variables:
        if iv.allowed_values is not None:
            raw_conditions[iv.name] = rng.choice(
                iv.allowed_values, size=num_samples, replace=replace
            )
        elif (iv.value_range is not None) and (iv.type == ValueType.REAL):
            raw_conditions[iv.name] = rng.uniform(*iv.value_range, size=num_samples)

        else:
            raise ValueError(
                "allowed_values or [value_range and type==REAL] needs to be set for "
                "%s" % (iv)
            )

    conditions = pd.DataFrame(raw_conditions)
    return Result(conditions=conditions)


@singledispatch
def random_sample(s, **kwargs):
    """Function to create a sequence of conditions randomly sampled from conditions."""
    raise NotImplementedError(
        "random_sample doesn't have an implementation for %s (type=%s)" % (s, type(s))
    )


@random_sample.register(State)
def random_sample_on_state(s: State, **kwargs) -> State:
    return wrap_to_use_state(random_sample_on_conditions)(s, **kwargs)


@random_sample.register(list)
@random_sample.register(tuple)
def random_sample_on_list(
    conditions: Union[list, tuple],
    num_samples: int = 1,
    random_state: Optional[int] = None,
    replace: bool = False,
) -> list:
    """
    Examples:
        >>> random_sample([1, 1, 2, 2, 3, 3], num_samples=2, random_state=1, replace=True)
        [1, 3]

        >>> random_sample((1, 1, 2, 2, 3, 3), num_samples=3, random_state=1, replace=True)
        [1, 3, 3]


    """

    if random_state is not None:
        random.seed(random_state)

    assert replace is True, "random.choices only supports choice with replacement."
    return random.choices(conditions, k=num_samples)


@random_sample.register(pd.DataFrame)
@random_sample.register(np.ndarray)
@random_sample.register(np.recarray)
def random_sample_on_conditions(
    conditions: Union[pd.DataFrame, np.ndarray, np.recarray],
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
        >>> random_sample(
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
