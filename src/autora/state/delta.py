"""Classes to represent cycle state $S$ as $S_n = S_{0} + \\sum_{i=1}^n \\Delta S_{i}"""
from __future__ import annotations

import dataclasses
import inspect
from collections import UserDict
from dataclasses import dataclass, fields, replace
from functools import singledispatch, wraps
from typing import Generic, List, TypeVar

import numpy as np
import pandas as pd

S = TypeVar("S")
T = TypeVar("T")


@dataclass(frozen=True)
class State:
    """
    Base object for dataclasses which use the Delta mechanism.

    Examples:
        >>> from dataclasses import dataclass, field

        We define a dataclass where each field (which is going to be delta-ed) has additional
        metadata "delta" which describes its delta behaviour.
        >>> @dataclass(frozen=True)
        ... class ListState(State):
        ...    l: List = field(default_factory=list, metadata={"delta": "extend"})
        ...    m: List = field(default_factory=list, metadata={"delta": "replace"})

        Now we instantiate the dataclass...
        >>> l = ListState(l=list("abc"), m=list("xyz"))
        >>> l
        ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        ... and can add deltas to it. `l` will be extended:
        >>> l + Delta(l=list("def"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['x', 'y', 'z'])

        ... wheras `m` will be replaced:
        >>> l + Delta(m=list("uvw"))
        ListState(l=['a', 'b', 'c'], m=['u', 'v', 'w'])

        ... they can be chained:
        >>> l + Delta(l=list("def")) + Delta(m=list("uvw"))
        ListState(l=['a', 'b', 'c', 'd', 'e', 'f'], m=['u', 'v', 'w'])

        ... and we update multiple fields with one Delta:
        >>> l + Delta(l=list("ghi"), m=list("rst"))
        ListState(l=['a', 'b', 'c', 'g', 'h', 'i'], m=['r', 's', 't'])

        Passing a nonexistent field will cause an error:
        >>> l + Delta(o="not a field")
        Traceback (most recent call last):
        ...
        AttributeError: key=`o` is missing on ListState(l=['a', 'b', 'c'], m=['x', 'y', 'z'])

        We can also use the `.update` method to do the same thing:
        >>> l.update(l=list("ghi"), m=list("rst"))
        ListState(l=['a', 'b', 'c', 'g', 'h', 'i'], m=['r', 's', 't'])

        We can also define fields which `append` the last result:
        >>> @dataclass(frozen=True)
        ... class AppendState(State):
        ...    n: List = field(default_factory=list, metadata={"delta": "append"})

        >>> m = AppendState(n=list("ɑβɣ"))
        >>> m
        AppendState(n=['ɑ', 'β', 'ɣ'])

        `n` will be appended:
        >>> m + Delta(n="∂")
        AppendState(n=['ɑ', 'β', 'ɣ', '∂'])

    """

    def __add__(self, other: Delta):
        updates = dict()
        for key, other_value in other.data.items():
            try:
                self_field = next(filter(lambda f: f.name == key, fields(self)))
            except StopIteration:
                raise AttributeError("key=`%s` is missing on %s" % (key, self))
            delta_behavior = self_field.metadata["delta"]
            self_value = getattr(self, key)
            if delta_behavior == "extend":
                extended_value = extend(self_value, other_value)
                updates[key] = extended_value
            elif delta_behavior == "append":
                appended_value = append(self_value, other_value)
                updates[key] = appended_value
            elif delta_behavior == "replace":
                updates[key] = other_value
            else:
                raise NotImplementedError(
                    "delta_behaviour=`%s` not implemented" % (delta_behavior)
                )

        new = replace(self, **updates)
        return new

    def update(self, **kwargs):
        return self + Delta(**kwargs)


class Delta(UserDict, Generic[S]):
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


Result = Delta
"""`Result` is an alias for `Delta`."""


@singledispatch
def extend(a, b):
    """
    Function to extend supported datatypes.

    """
    raise NotImplementedError("`extend` not implemented for %s, %s" % (a, b))


@extend.register(list)
def extend_list(a, b):
    """
    Examples:
        >>> extend([], [])
        []

        >>> extend([1,2], [3])
        [1, 2, 3]
    """
    return a + b


@extend.register(pd.DataFrame)
def extend_pd_dataframe(a, b):
    """
    Examples:
        >>> extend(pd.DataFrame({"a": []}), pd.DataFrame({"a": []}))
        Empty DataFrame
        Columns: [a]
        Index: []

        >>> extend(pd.DataFrame({"a": [1,2,3]}), pd.DataFrame({"a": [4,5,6]}))
           a
        0  1
        1  2
        2  3
        3  4
        4  5
        5  6
    """
    return pd.concat((a, b), ignore_index=True)


def append(a: List[T], b: T) -> List[T]:
    # TODO: add DOCTESTS
    return a + [b]


@extend.register(np.ndarray)
def extend_np_ndarray(a, b):
    """
    Examples:
        >>> extend(np.array([(1,2,3), (4,5,6)]), np.array([(7,8,9)]))
        array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
    """
    return np.row_stack([a, b])


@extend.register(dict)
def extend_dict(a, b):
    """
    Examples:
        >>> extend({"a": "cats"}, {"b": "dogs"})
        {'a': 'cats', 'b': 'dogs'}
    """
    return dict(a, **b)


def wrap_to_use_state(f):
    """Decorator to make target `f` into a function on a `State` and `**kwargs`.

    This wrapper makes it easier to pass arguments to a function from a State.

    It was inspired by the pytest "fixtures" mechanism.

    Args:
        f:

    Returns:

    Examples:
        >>> from autora.state.delta import State, Delta
        >>> from dataclasses import dataclass, field
        >>> import pandas as pd
        >>> from typing import List, Optional

        The `State` it operates on needs to have the metadata described in the state module:
        >>> @dataclass(frozen=True)
        ... class S(State):
        ...     conditions: List[int] = field(metadata={"delta": "replace"})

        We indicate the inputs required by the parameter names.
        The output must be a `Delta` object.
        >>> from autora.state.delta import Delta
        >>> @wrap_to_use_state
        ... def experimentalist(conditions):
        ...     new_conditions = [c + 10 for c in conditions]
        ...     return Delta(conditions=new_conditions)

        >>> experimentalist(S(conditions=[1,2,3,4]))
        S(conditions=[11, 12, 13, 14])

        >>> experimentalist(S(conditions=[101,102,103,104]))
        S(conditions=[111, 112, 113, 114])

        >>> from autora.variable import VariableCollection, Variable
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.linear_model import LinearRegression

        >>> @wrap_to_use_state
        ... def theorist(experiment_data: pd.DataFrame, variables: VariableCollection, **kwargs):
        ...     ivs = [v.name for v in variables.independent_variables]
        ...     dvs = [v.name for v in variables.dependent_variables]
        ...     X, y = experiment_data[ivs], experiment_data[dvs]
        ...     new_model = LinearRegression(fit_intercept=True).set_params(**kwargs).fit(X, y)
        ...     return Delta(model=new_model)

        >>> @dataclass(frozen=True)
        ... class T(State):
        ...     variables: VariableCollection  # field(metadata={"delta":... }) omitted ∴ immutable
        ...     experiment_data: pd.DataFrame = field(metadata={"delta": "extend"})
        ...     model: Optional[BaseEstimator] = field(metadata={"delta": "replace"}, default=None)

        >>> t = T(
        ...     variables=VariableCollection(independent_variables=[Variable("x")],
        ...                                  dependent_variables=[Variable("y")]),
        ...     experiment_data=pd.DataFrame({"x": [0,1,2,3,4], "y": [2,3,4,5,6]})
        ... )
        >>> t_prime = theorist(t)
        >>> t_prime.model.coef_, t_prime.model.intercept_
        (array([[1.]]), array([2.]))

        Arguments from the state can be overridden by passing them in as keyword arguments (kwargs):
        >>> theorist(t, experiment_data=pd.DataFrame({"x": [0,1,2,3], "y": [12,13,14,15]}))\\
        ...     .model.intercept_
        array([12.])

        ... and other arguments supported by the inner function can also be passed
        (if and only if the inner function allows for and handles `**kwargs` arguments alongside
        the values from the state).
        >>> theorist(t, fit_intercept=False).model.intercept_
        0.0

        Any parameters not provided by the state must be provided by default values or by the
        caller. If the default is specified:
        >>> @wrap_to_use_state
        ... def experimentalist(conditions, offset=25):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then it need not be passed.
        >>> experimentalist(S(conditions=[1,2,3,4]))
        S(conditions=[26, 27, 28, 29])

        If a default isn't specified:
        >>> @wrap_to_use_state
        ... def experimentalist(conditions, offset):
        ...     new_conditions = [c + offset for c in conditions]
        ...     return Delta(conditions=new_conditions)

        ... then calling the experimentalist without it will throw an error:
        >>> experimentalist(S(conditions=[1,2,3,4]))
        Traceback (most recent call last):
        ...
        TypeError: experimentalist() missing 1 required positional argument: 'offset'

        ... which can be fixed by passing the argument as a keyword to the wrapped function.
        >>> experimentalist(S(conditions=[1,2,3,4]), offset=2)
        S(conditions=[3, 4, 5, 6])

    """
    # Get the set of parameter names from function f's signature
    parameters_ = set(inspect.signature(f).parameters.keys())

    @wraps(f)
    def _f(state_: S, /, **kwargs) -> S:
        # Get the parameters needed which are available from the state_.
        # All others must be provided as kwargs or default values on f.
        assert dataclasses.is_dataclass(state_)
        from_state = parameters_.intersection(
            {i.name for i in dataclasses.fields(state_)}
        )
        arguments_from_state = {k: getattr(state_, k) for k in from_state}
        arguments = dict(arguments_from_state, **kwargs)
        delta = f(**arguments)
        new_state = state_ + delta
        return new_state

    return _f
