"""Classes to represent cycle state $S$ as $S_n = S_{0} + \\sum_{i=1}^n \\Delta S_{i}"""
from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass, fields, replace
from functools import singledispatch
from typing import Generic, TypeVar

import numpy as np
import pandas as pd

S = TypeVar("S")


@dataclass(frozen=True)
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
            elif delta_behavior == "replace":
                updates[key] = other_value
            else:
                raise NotImplementedError(
                    "delta_behaviour=`%s` not implemented" % (delta_behavior)
                )

        new = replace(self, **updates)
        return new


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


@singledispatch
def extend(a: S, b: S) -> S:
    """
    Function to extend supported datatypes.

    Examples:
        >>> extend([], [])
        []

        >>> extend([1,2], [3])
        [1, 2, 3]

    """
    raise NotImplementedError("`extend` not implemented for %s, %s" % (a, b))


@extend.register
def extend_list(a: list, b: list) -> list:
    """
    Examples:
        >>> extend([], [])
        []

        >>> extend([1,2], [3])
        [1, 2, 3]
    """
    return a + b


@extend.register
def extend_pd_dataframe(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
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


@extend.register
def extend_np_ndarray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Examples:
        >>> extend(np.array([(1,2,3), (4,5,6)]), np.array([(7,8,9)]))
        array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
    """
    return np.row_stack([a, b])


@extend.register
def extend_dict(a: dict, b: dict) -> dict:
    """
    Examples:
        >>> extend({"a": "cats"}, {"b": "dogs"})
        {'a': 'cats', 'b': 'dogs'}
    """
    return dict(a, **b)
