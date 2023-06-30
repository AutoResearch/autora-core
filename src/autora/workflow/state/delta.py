"""Classes to represent cycle state $S$ as $S_n = S_{0} + \sum_{i=1}^n \Delta S_{i}"""
from __future__ import annotations

import dataclasses
import inspect
from collections import UserDict
from functools import wraps
from typing import Generic, Literal, Optional, TypeVar, Union

import numpy as np
import pandas as pd

S = TypeVar("S")

from dataclasses import dataclass, field, fields, replace
from typing import List


class BaseState:
    """
    BaseState for dataclasses which use the Delta mechanism.

    Examples:
        >>> from dataclasses import dataclass, field

        We define a dataclass where each field (which is going to be delta-ed) has additional
        metadata "delta" which describes its delta behaviour.
        >>> @dataclass(frozen=True)
        ... class ListState(BaseState):
        ...    l: List = field(default_factory=list, metadata={"delta": "extend"})
        ...    m: List = field(default_factory=list, metadata={"delta": "replace"})

        Now we instantiate the dataclass...
        >>> l = ListState(l=list("abc"), m=list("xyz"))
        >>> l
        ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... and can add deltas to it. `l` will be extended:
        >>> l + BaseDelta(l=list("def"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['x', 'y', 'z'])

        ... wheras `m` will be replaced:
        >>> l + BaseDelta(m=list("uvw"))
        ListState(l=['a', 'b', 'c'], m=['u', 'v', 'w'])

        ... they can be chained:
        >>> l + BaseDelta(l=list("def")) + BaseDelta(m=list("uvw"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['u', 'v', 'w'])

        ... and we update multiple fields with one Delta:
        >>> l + BaseDelta(l=list("ghi"), m=list("rst"))
        ListState(l=['a', 'b', 'c', 'g', 'h', 'i'], m=['r', 's', 't'])

        Passing a nonexistent field will cause an error:
        >>> l + BaseDelta(o="not a field")
        Traceback (most recent call last):
        ...
        AttributeError: key=`o` is missing on ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

    """

    def __add__(self, other: BaseDelta):
        updates = dict()
        for key, other_value in other.data.items():
            try:
                self_field = next(filter(lambda f: f.name == key, fields(self)))
            except StopIteration:
                raise AttributeError("key=`%s` is missing on %s" % (key, self))
            delta_behavior = self_field.metadata["delta"]
            self_value = getattr(self, key)
            if delta_behavior == "extend":
                extended_value = _get_extended_value(self_value, other_value)
                updates[key] = extended_value
            elif delta_behavior == "replace":
                updates[key] = other_value
            else:
                raise NotImplementedError(
                    "delta_behaviour=`%s` not implemented" % (delta_behavior)
                )

        new = replace(self, **updates)
        return new


class BaseDelta(UserDict, Generic[S]):
    """
    Represents a delta where the base object determines the extension behavior.

    Examples:
        >>> from dataclasses import dataclass

        First we define the dataclass to act as the basis:
        >>> from typing import Optional, List
        >>> @dataclass(frozen=True)
        ... class ListState:
        ...     l: Optional[List] = None
        ...     m: Optional[List] = None
        ...
    """

    pass


def _get_extended_value(base, extension):
    if isinstance(base, list):
        assert isinstance(extension, list)
        return base + extension
    elif isinstance(base, pd.DataFrame):
        return pd.concat((base, extension), ignore_index=True)
    elif isinstance(base, np.ndarray):
        return np.row_stack([base, extension])
    elif isinstance(base, dict):
        return dict(base, **extension)


def wrap_to_use_state(f):
    """

    Args:
        f:

    Returns:

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class S(BaseState):
        ...     conditions: list[int] = field(metadata={"delta": "replace"})

        >>> @wrap_to_use_state
        ... def function(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return BaseDelta(conditions=new_conditions)

        >>> function(S(conditions=[1,2,3,4]))
        S(conditions=[11, 12, 13, 14])

        >>> function(S(conditions=[101,102,103,104]))
        S(conditions=[111, 112, 113, 114])

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        >>> @wrap_to_use_state
        ... def function(experimental_data: pd.DataFrame, variables: VariableCollection):
        ...     ivs = [v.name for v in variables.independent_variables]
        ...     dvs = [v.name for v in variables.dependent_variables]
        ...     X, y = experimental_data[ivs], experimental_data[dvs]
        ...     new_model = LinearRegression(fit_intercept=True).fit(X, y)
        ...     return BaseDelta(model=new_model)

        >>> @dataclass(frozen=True)
        ... class T(BaseState):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experimental_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> t = T(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experimental_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )
        >>> t_prime = function(t)
        >>> t_prime.model.coef_, t_prime.model.intercept_
        (array([[1.]]), array([2.]))
    """
    parameters = inspect.signature(f).parameters

    @wraps(f)
    # TODO: how do we handle the params here?
    def _f(state: S, params: Optional[dict] = None) -> S:
        # Convert the dataclass to a dict of parameters
        arguments = dict((p, getattr(state, p)) for p in parameters)
        delta = f(**arguments)
        new_state = state + delta
        return new_state

    return _f
