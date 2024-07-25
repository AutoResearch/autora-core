from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Mapping, Union

from autora.state import Delta


@dataclass(frozen=True)
class Entry:
    kind: str
    data: Any


@dataclass(frozen=True)
class HistoryState:
    """

    Examples:
        >>> a = HistoryState()
        >>> a
        HistoryState(data=[])

        >>> b = a + {"conditions": [1, 2, 3]}
        >>> b
        HistoryState(data=[Entry(kind='conditions', data=[1, 2, 3])])

        >>> list(b.filter_by("conditions"))
        [Entry(kind='conditions', data=[1, 2, 3])]

    """

    data: List[Entry] = field(default_factory=list)

    def __add__(self, other: Union[Delta, Mapping]):
        new_data = [Entry(kind, data) for kind, data in other.items()]
        new = replace(self, data=self.data + new_data)
        return new

    def filter_by(self, kind):
        filtered_data = filter(lambda d: d.kind == kind, self.data)
        return filtered_data


@dataclass(frozen=True)
class StandardHistoryState(HistoryState):
    """

    Examples:
        >>> a = StandardHistoryState()
        >>> a.conditions
        []

        >>> b = a + {"conditions": [1, 2, 3]}
        >>> b.conditions
        [[1, 2, 3]]

        `a` is still empty.
        >>> a.conditions
        []

        >>> c = b + {"conditions": [4, 5]}
        >>> c.conditions
        [[1, 2, 3], [4, 5]]

        >>> c  # doctest: +NORMALIZE_WHITESPACE
        StandardHistoryState(data=[Entry(kind='conditions', data=[1, 2, 3]),
                                   Entry(kind='conditions', data=[4, 5])])

        >>> c.experiment_data
        []

        >>> c.nonexistent_key
        Traceback (most recent call last):
        ...
        AttributeError: 'StandardHistoryState' object has no attribute 'nonexistent_key'


    """

    @property
    def conditions(self):
        return [d.data for d in self.filter_by("conditions")]

    @property
    def experiment_data(self):
        return [d.data for d in self.filter_by("experiment_data")]

    @property
    def model(self):
        return [d.data for d in self.filter_by("model")]

    models = model

    @property
    def variables(self):
        return [d.data for d in self.filter_by("variables")]
