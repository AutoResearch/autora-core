"""Classes to represent cycle state $S$ as $S_n = S_{0} + \sum_{i=1}^n \Delta S_{i}"""
from __future__ import annotations

import dataclasses
import inspect
from functools import wraps
from typing import Generic, Literal, Optional, TypeVar, Union

import numpy as np
import pandas as pd

S = TypeVar("S")


@dataclasses.dataclass(frozen=True)
class Delta(Generic[S]):
    """Representing a delta on top of a dataclass.

    Examples:
        >>> from dataclasses import dataclass

        Use the Delta to handle updates to a state containing two lists.

        First we define the dataclass to act as the basis:
        >>> from typing import Optional, List
        >>> @dataclass(frozen=True)
        ... class ListState:
        ...     l: Optional[List] = None
        ...     m: Optional[List] = None

        Next we define the delta dataclass – this inherits its fields from the basis, and inherits
        the delta-logic from the `Delta` class. It is important that the types of this object
        default to "None" if they are not set.
        >>> @dataclass(frozen=True)
        ... class ListStateDelta(ListState, Delta):
        ...     pass

        We start with an emtpy list-state:
        >>> s = ListState(l=[], m=[])
        >>> s
        ListState(l=[], m=[])

        We can extend the state (extending each of the lists independently) by adding a delta:
        >>> d = ListStateDelta(kind="extend", l=["a"], m=list("abcde"))
        >>> s + d
        ListState(l=['a'], m=['a', 'b', 'c', 'd', 'e'])

        ... or adding multiple deltas at once:
        >>> e = ListStateDelta(kind="extend", l=list("bc"), m=["f"])
        >>> s + d + e
        ListState(l=['a', 'b', 'c'], m=['a', 'b', 'c', 'd', 'e', 'f'])

        >>> s + e
        ListState(l=['b', 'c'], m=['f'])

        We can also make a "replace" delta which replaces a particular field but leaves any others
        unchanged:
        >>> f = ListStateDelta(kind="replace", l=list("lace"))
        >>> s + e + f
        ListState(l=['l', 'a', 'c', 'e'], m=['f'])

        Use the Delta to handle updates to a state containing a dataframe:
        >>> import pandas as pd
        >>> @dataclass(frozen=True)
        ... class DataFrameState:
        ...     data: pd.DataFrame

        >>> @dataclass(frozen=True)
        ... class DataFrameStateDelta(DataFrameState, Delta):
        ...     pass

        >>> s = DataFrameState(data=pd.DataFrame({"a": [1], "b": ["f"]}))
        >>> s.data
           a  b
        0  1  f

        >>> d = DataFrameStateDelta(kind="extend", data=pd.DataFrame({"a":[2], "b":["s"]}))
        >>> (s + d).data
           a  b
        0  1  f
        1  2  s

    """

    kind: Literal["extend", "replace"]

    def __radd__(self, other: Union[S, Delta]):
        updates = dict()
        for f in dataclasses.fields(other):
            if (value := getattr(self, f.name)) is not None:
                if self.kind == "replace":
                    updates[f.name] = value
                elif self.kind == "extend":
                    other_value = getattr(other, f.name)
                    extended_value = _get_extended_value(other_value, value)
                    updates[f.name] = extended_value
        new = dataclasses.replace(other, **updates)
        return new


class GeneralDelta(Generic[S]):
    """Representing a delta on top of a dataclass.

    Examples:
        >>> from dataclasses import dataclass

        Use the Delta to handle updates to a state containing two lists.

        First we define the dataclass to act as the basis:
        >>> from typing import Optional, List
        >>> @dataclass(frozen=True)
        ... class ListState:
        ...     l: Optional[List] = None
        ...     m: Optional[List] = None

        We start with an emtpy list-state:
        >>> s = ListState(l=[], m=[])
        >>> s
        ListState(l=[], m=[])

        We can extend the state (extending each of the lists independently) by adding a delta:
        >>> d = GeneralDelta(kind="extend", l=["a"], m=list("abcde"))
        >>> s + d
        ListState(l=['a'], m=['a', 'b', 'c', 'd', 'e'])

        ... or adding multiple deltas at once:
        >>> e = GeneralDelta(kind="extend", l=list("bc"), m=["f"])
        >>> s + d + e
        ListState(l=['a', 'b', 'c'], m=['a', 'b', 'c', 'd', 'e', 'f'])

        >>> s + e
        ListState(l=['b', 'c'], m=['f'])

        We can also make a "replace" delta which replaces a particular field but leaves any others
        unchanged:
        >>> f = GeneralDelta(kind="replace", l=list("lace"))
        >>> s + e + f
        ListState(l=['l', 'a', 'c', 'e'], m=['f'])

        Use the Delta to handle updates to a state containing a dataframe:
        >>> import pandas as pd
        >>> @dataclass(frozen=True)
        ... class DataFrameState:
        ...     data: pd.DataFrame


        >>> s = DataFrameState(data=pd.DataFrame({"a": [1], "b": ["f"]}))
        >>> s.data
           a  b
        0  1  f

        >>> d = GeneralDelta(kind="extend", data=pd.DataFrame({"a":[2], "b":["s"]}))
        >>> (s + d).data
           a  b
        0  1  f
        1  2  s

    """

    def __init__(self, kind: Optional[Literal["extend", "replace"]] = None, **kwargs):
        self.kind = kind
        self.data: dict = kwargs

    def __radd__(self, other):
        if self.kind is None:
            return other

        other_fields = set(field.name for field in dataclasses.fields(other))

        updates = dict()
        for key, value in self.data.items():
            assert key in other_fields, f"{key=} must be in the left dataclass"
            assert value is not None, f"{value=} may not be None"

            if self.kind == "replace":
                updates[key] = value
            elif self.kind == "extend":
                other_value = getattr(other, key)
                extended_value = _get_extended_value(other_value, value)
                updates[key] = extended_value

        new = dataclasses.replace(other, **updates)
        return new


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


@dataclasses.dataclass(frozen=True)
class State:
    data: Optional[Union[pd.DataFrame, np.typing.ArrayLike]]


@dataclasses.dataclass(frozen=True)
class StateDelta(State, Delta):
    """"""

    pass


def wrap_to_use_state(f):
    """

    Args:
        f:

    Returns:

    Examples:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class S:
        ...     conditions: list[int]

        >>> @wrap_to_use_state
        ... def function(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return GeneralDelta("replace", conditions=new_conditions)

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
        ...     return GeneralDelta("replace", model=new_model)

        >>> @dataclass
        ... class T:
        ...     variables: VariableCollection
        ...     experimental_data: pd.DataFrame
        ...     model: Optional[BaseEstimator] = None

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
