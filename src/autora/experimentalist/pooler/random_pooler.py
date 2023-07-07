import random
from itertools import product
from typing import Iterable, List, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from autora.state.delta import Result, wrap_to_use_state
from autora.utils.deprecation import deprecated_alias
from autora.variable import IV, ValueType, Variable, VariableCollection


@wrap_to_use_state
def random_experimentalist(
    variables: VariableCollection,
    num_samples=5,
    random_state=None,
    fmt: Type = pd.DataFrame,
    duplicates: bool = True,
):
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
        >>> random_experimentalist(s, random_state=1).conditions
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
        >>> random_experimentalist(t, random_state=1).conditions
                  x
        0  0.118216
        1  4.504637
        2 -3.558404
        3  4.486494
        4 -1.881685



        The allowed_values or value_range must be specified:
        >>> random_experimentalist(
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
        >>> random_experimentalist(t, num_samples=10, duplicates=True, random_state=1).conditions
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
        >>> random_experimentalist(S(
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
        >>> random_experimentalist(u, random_state=1).conditions
             x  y     z
        0 -0.6  3  29.0
        1  0.2  4  24.0
        2  5.2  4  23.0
        3  9.0  3  29.0
        4 -9.4  3  22.0

        The output can be in several formats. The default is pd.DataFrame.
        Alternative: `np.recarray`:
        >>> random_experimentalist(s, fmt=np.recarray, random_state=1).conditions
        rec.array([(4,), (5,), (7,), (9,), (0,)],
                  dtype=[('x', '<i...')])

        >>> random_experimentalist(t, fmt=np.recarray, random_state=1).conditions
        rec.array([(2,  72), (3, 411), (4, 474), (4, 125), (1, 156)],
                  dtype=[('x1', '<i...'), ('x2', '<i...')])

        Alternative: `np.array` (without field names):
        >>> random_experimentalist(t, fmt=np.array, random_state=1).conditions
        array([[  2,  72],
               [  3, 411],
               [  4, 474],
               [  4, 125],
               [  1, 156]])

    """
    rng = np.random.default_rng(random_state)

    raw_conditions = {}
    for iv in variables.independent_variables:
        if iv.allowed_values is not None:
            raw_conditions[iv.name] = rng.choice(
                iv.allowed_values, size=num_samples, replace=duplicates
            )
        elif (iv.value_range is not None) and (iv.type == ValueType.REAL):
            raw_conditions[iv.name] = rng.uniform(*iv.value_range, size=num_samples)

        else:
            raise ValueError(
                "allowed_values or [value_range and type==REAL] needs to be set for "
                "%s" % (iv)
            )

    iv_names = [v.name for v in variables.independent_variables]
    if fmt is pd.DataFrame:
        conditions = pd.DataFrame(raw_conditions)
    elif fmt is np.recarray:
        conditions = np.core.records.fromarrays(
            [raw_conditions[n] for n in iv_names], names=iv_names
        )  # type: ignore
    elif fmt is np.array:
        conditions = np.column_stack([raw_conditions[n] for n in iv_names])
    else:
        raise NotImplementedError("fmt=%s is not supported" % (fmt))

    return Result(conditions=conditions)


def grid_pool(ivs: Sequence[Variable]):
    """
    Pipeline function to create an exhaustive pool from discrete values
    using a Cartesian product of sets.
    """
    # Get allowed values for each IV
    l_iv_values = []
    for iv in ivs:
        assert iv.allowed_values is not None, (
            f"grid_pool requires allowed_values to be set, "
            f"but allowed_values is None on {iv=} "
        )
        l_iv_values.append(iv.allowed_values)

    # Return Cartesian product of all IV values
    return product(*l_iv_values)


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
