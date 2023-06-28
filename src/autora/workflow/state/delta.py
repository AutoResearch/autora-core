"""Classes to represent cycle state $S$ as $S_n = S_{0} + \sum_{i=1}^n \Delta S_{i}"""
from __future__ import annotations

import dataclasses
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

        First we define the dataclass *with optional types* to act as the basis:
        >>> from typing import Optional, List
        >>> @dataclass(frozen=True)
        ... class ListState:
        ...     l: Optional[List] = None
        ...     m: Optional[List] = None

        Next we define the delta dataclass – this inherits its fields from the basis, and inherits
        the delta-logic from the `Delta` class.
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

        We can also make a delta which replaces a particular field but leaves any others:
        >>> f = ListStateDelta(kind="replace", l=list("repl"))
        >>> s + e + f
        ListState(l=['r', 'e', 'p', 'l'], m=['f'])

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
                    if isinstance(other_value, list):
                        assert isinstance(value, list)
                        updates[f.name] = other_value + value
                    elif isinstance(other_value, pd.DataFrame):
                        updates[f.name] = pd.concat(
                            (other_value, value), ignore_index=True
                        )
                    elif isinstance(other_value, np.ndarray):
                        updates[f.name] = np.row_stack([other_value, value])
                    elif isinstance(other_value, dict):
                        updates[f.name] = dict(other_value, **value)
        new = dataclasses.replace(other, **updates)
        return new


@dataclasses.dataclass(frozen=True)
class State:
    data: Optional[Union[pd.DataFrame, np.typing.ArrayLike]]


@dataclasses.dataclass(frozen=True)
class StateDelta(State, Delta):
    """"""

    pass
